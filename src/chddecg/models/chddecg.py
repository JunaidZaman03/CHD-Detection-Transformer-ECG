import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from .resnet_module import InputConv, ResBlock, SE
from .transformer_module import MultiHeadAttention, EncoderLayer, TemporalAttention
from .tabnet.custom_objects import glu, sparsemax, GroupNormalization, custom_objects, AssertFiniteLayer
from .tabnet_downsampling import TabNet_downsampling

def CHDdECG(num_classes=2, use_tabnet=True, use_attention=True):
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")

    input_signal = tf.keras.Input(shape=(5000, 12), dtype=tf.float16, name='signal_input')
    input_clinical = tf.keras.Input(shape=(15,), dtype=tf.float16, name='clinical_input')
    input_wavelet = tf.keras.Input(shape=(100,), dtype=tf.float16, name='wavelet_input')

    expected_signal_shape = (None, 5000, 12)
    expected_clinical_shape = (None, 15)
    expected_wavelet_shape = (None, 100)

    if input_signal.shape[1:] != expected_signal_shape[1:]:
        raise ValueError(f"Expected signal_input shape {expected_signal_shape}, got {input_signal.shape}")
    if input_clinical.shape[1:] != expected_clinical_shape[1:]:
        raise ValueError(f"Expected clinical_input shape {expected_clinical_shape}, got {input_clinical.shape}")
    if input_wavelet.shape[1:] != expected_wavelet_shape[1:]:
        raise ValueError(f"Expected wavelet_input shape {expected_wavelet_shape}, got {input_wavelet.shape}")

    signal_checked = AssertFiniteLayer(name='signal_input_check')(input_signal)
    clinical_checked = AssertFiniteLayer(name='clinical_input_check')(input_clinical)
    wavelet_checked = AssertFiniteLayer(name='wavelet_input_check')(input_wavelet)

    Signal_block_conv = InputConv(
        filter_num=24,
        kernel_size=30,
        stride=2,
        name='signal_block_conv'
    )(signal_checked)

    Signal_block_res2_1 = ResBlock(filter_num=12, kernel_size=3, stride=2, name='signal_block_res2_1')(Signal_block_conv)
    Signal_block_res3_1 = ResBlock(filter_num=12, kernel_size=3, stride=2, name='signal_block_res3_1')(Signal_block_res2_1)
    Signal_block_res4_1 = ResBlock(filter_num=12, kernel_size=3, stride=2, name='signal_block_res4_1')(Signal_block_res3_1)

    Signal_block_res2_2 = ResBlock(filter_num=12, kernel_size=5, stride=2, name='signal_block_res2_2')(Signal_block_conv)
    Signal_block_res3_2 = ResBlock(filter_num=12, kernel_size=5, stride=2, name='signal_block_res3_2')(Signal_block_res2_2)
    Signal_block_res4_2 = ResBlock(filter_num=12, kernel_size=5, stride=2, name='signal_block_res4_2')(Signal_block_res3_2)

    Signal_block_res2_3 = ResBlock(filter_num=12, kernel_size=7, stride=2, name='signal_block_res2_3')(Signal_block_conv)
    Signal_block_res3_3 = ResBlock(filter_num=12, kernel_size=7, stride=2, name='signal_block_res3_3')(Signal_block_res2_3)
    Signal_block_res4_3 = ResBlock(filter_num=12, kernel_size=7, stride=2, name='signal_block_res4_3')(Signal_block_res3_3)

    model_con = layers.Concatenate(axis=-1, name='conv_concat')([
        Signal_block_res4_1, Signal_block_res4_2, Signal_block_res4_3
    ])

    model_con_se = SE(filter_sq=18, input_channel=36, name='signal_se')(model_con)

    model_con_downsampled = layers.MaxPooling1D(pool_size=4, strides=4, name='signal_downsample')(model_con_se)

    if use_attention:
        d_model = 36
        transformer_block = EncoderLayer(
            d_model=d_model,
            num_heads=6,
            dff=96,
            dropout_rate=0.3,
            name='signal_block_trans'
        )(model_con_downsampled)

        Signal_temporal_attention = TemporalAttention(
            feature_dim=d_model,
            kernel_size=3,
            stride=1,
            dropout_rate=0.2,
            use_residual=True,
            name='signal_tem_atten'
        )(transformer_block)
    else:
        Signal_temporal_attention = layers.Conv1D(
            filters=36,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='signal_conv_fallback',
            kernel_initializer='he_normal',
            dtype=tf.float16,
            kernel_regularizer=l2(0.05)
        )(model_con_downsampled)

    signal_pooled = layers.GlobalAveragePooling1D(name='signal_pool', dtype=tf.float16)(Signal_temporal_attention)

    signal_pre = layers.Dense(
        48, name='signal_pre', kernel_initializer='he_normal',
        kernel_regularizer=l2(0.05)
    )(signal_pooled)
    signal_pre = layers.Dropout(0.4)(signal_pre)

    if use_tabnet:
        Signal_block_tab1 = TabNet_downsampling(
            num_features=48,
            feature_dim=24,
            output_dim=24,
            num_decision_steps=2,
            relaxation_factor=1.5,
            sparsity_coefficient=5e-4,
            norm_type='group',
            name='signal_block_tab'
        )(signal_pre)
    else:
        Signal_block_tab1 = layers.Dense(
            24, activation='relu', name='signal_tab_fallback',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        )(signal_pre)

    if use_tabnet:
        clinical_pre = layers.Dense(
            12, activation='relu', name='clinical_pre',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        )(clinical_checked)
        clinical_pre = layers.Dropout(0.4)(clinical_pre)
        Clinical_block_tab1 = TabNet_downsampling(
            num_features=12,
            feature_dim=6,
            output_dim=12,
            num_decision_steps=1,
            relaxation_factor=1.5,
            sparsity_coefficient=5e-4,
            norm_type='group',
            name='clinical_block_tab'
        )(clinical_pre)
    else:
        Clinical_block_tab1 = layers.Dense(
            12, activation='relu', name='clinical_fallback',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        )(clinical_checked)

    if use_tabnet:
        Wavelet_block_tab1 = TabNet_downsampling(
            num_features=100,
            feature_dim=24,
            output_dim=12,
            num_decision_steps=1,
            relaxation_factor=1.5,
            sparsity_coefficient=5e-4,
            norm_type='group',
            name='wavelet_block_tab'
        )(wavelet_checked)
        Wavelet_block_tab1 = layers.Dense(
            6, activation='relu', name='wavelet_projection',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        )(Wavelet_block_tab1)
        Wavelet_block_tab1 = layers.Dropout(0.4)(Wavelet_block_tab1)
    else:
        Wavelet_block_tab1 = layers.Dense(
            6, activation='relu', name='wavelet_dense',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        )(wavelet_checked)

    fusion_block = layers.Concatenate(axis=-1, name='fusion_concat')([
        Signal_block_tab1, Clinical_block_tab1, Wavelet_block_tab1
    ])

    if use_tabnet:
        tab_all = TabNet_downsampling(
            num_features=42,
            feature_dim=24,
            output_dim=24,
            num_decision_steps=1,
            relaxation_factor=1.5,
            sparsity_coefficient=5e-4,
            norm_type='group',
            name='tab_con'
        )(fusion_block)
    else:
        tab_all = layers.Dense(
            24, activation='relu', name='fusion_dense',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        )(fusion_block)

    fusion_dense1 = layers.Dense(
        12, activation='relu', name='fusion_dense1',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(0.05)
    )(tab_all)
    fusion_batch_norm1 = layers.BatchNormalization(
        epsilon=1e-6, name='fusion_batch_norm1'
    )(fusion_dense1)
    fusion_dropout1 = layers.Dropout(rate=0.4, name='fusion_dropout1')(fusion_batch_norm1)

    fusion_dense2 = layers.Dense(
        6, activation='relu', name='fusion_dense2',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(0.05)
    )(fusion_dropout1)
    fusion_batch_norm2 = layers.BatchNormalization(
        epsilon=1e-6, name='fusion_batch_norm2'
    )(fusion_dense2)
    fusion_dropout2 = layers.Dropout(rate=0.3, name='fusion_dropout2')(fusion_batch_norm2)

    if num_classes == 2:
        output = layers.Dense(
            1, activation='sigmoid', name='output',
            kernel_initializer='glorot_normal',
            dtype='float32'
        )(fusion_dropout2)
    else:
        output = layers.Dense(
            num_classes, activation='softmax', name='output',
            kernel_initializer='glorot_normal',
            dtype='float32'
        )(fusion_dropout2)

    model = tf.keras.Model(
        inputs=[input_signal, input_clinical, input_wavelet],
        outputs=output,
        name='CHDdECG'
    )

    return model
