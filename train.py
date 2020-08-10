from __future__ import absolute_import, division, print_function
import tensorflow as tf
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    EPOCHS, BATCH_SIZE, save_model_dir, model_index, save_every_n_epoch
from prepare_data import generate_datasets, load_and_preprocess_image
import math
from models import mobilenet_v1, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, \
    efficientnet, resnext, inception_v4, inception_resnet_v1, inception_resnet_v2, \
    se_resnet, squeezenet, densenet, shufflenet_v2, resnet, se_resnext
import optimization.distribution_utils as distribution_utils
from optimization.distribution_utils import broadcast_variables
from optimization.compression import Compression
from absl import logging

def get_model():
    if model_index == 0:
        return mobilenet_v1.MobileNetV1()
    elif model_index == 1:
        return mobilenet_v2.MobileNetV2()
    elif model_index == 2:
        return mobilenet_v3_large.MobileNetV3Large()
    elif model_index == 3:
        return mobilenet_v3_small.MobileNetV3Small()
    elif model_index == 4:
        return efficientnet.efficient_net_b0()
    elif model_index == 5:
        return efficientnet.efficient_net_b1()
    elif model_index == 6:
        return efficientnet.efficient_net_b2()
    elif model_index == 7:
        return efficientnet.efficient_net_b3()
    elif model_index == 8:
        return efficientnet.efficient_net_b4()
    elif model_index == 9:
        return efficientnet.efficient_net_b5()
    elif model_index == 10:
        return efficientnet.efficient_net_b6()
    elif model_index == 11:
        return efficientnet.efficient_net_b7()
    elif model_index == 12:
        return resnext.ResNeXt50()
    elif model_index == 13:
        return resnext.ResNeXt101()
    elif model_index == 14:
        return inception_v4.InceptionV4()
    elif model_index == 15:
        return inception_resnet_v1.InceptionResNetV1()
    elif model_index == 16:
        return inception_resnet_v2.InceptionResNetV2()
    elif model_index == 17:
        return se_resnet.se_resnet_50()
    elif model_index == 18:
        return se_resnet.se_resnet_101()
    elif model_index == 19:
        return se_resnet.se_resnet_152()
    elif model_index == 20:
        return squeezenet.SqueezeNet()
    elif model_index == 21:
        return densenet.densenet_121()
    elif model_index == 22:
        return densenet.densenet_169()
    elif model_index == 23:
        return densenet.densenet_201()
    elif model_index == 24:
        return densenet.densenet_264()
    elif model_index == 25:
        return shufflenet_v2.shufflenet_0_5x()
    elif model_index == 26:
        return shufflenet_v2.shufflenet_1_0x()
    elif model_index == 27:
        return shufflenet_v2.shufflenet_1_5x()
    elif model_index == 28:
        return shufflenet_v2.shufflenet_2_0x()
    elif model_index == 29:
        return resnet.resnet_18()
    elif model_index == 30:
        return resnet.resnet_34()
    elif model_index == 31:
        return resnet.resnet_50()
    elif model_index == 32:
        return resnet.resnet_101()
    elif model_index == 33:
        return resnet.resnet_152()
    elif model_index == 34:
        return se_resnext.SEResNeXt50()
    elif model_index == 35:
        return se_resnext.SEResNeXt101()
    else:
        raise ValueError("The model_index does not exist.")


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def process_features(features, data_augmentation):
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()

    return images, labels


if __name__ == '__main__':
    # GPU settings
    import byteps.tensorflow as hvd
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    
    if hvd.local_rank() == 0:
        logging.set_verbosity(logging.INFO)
    else:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
        logging.set_verbosity(logging.WARN)
    logging.info("Total workers: {}, local workers: {}".format(
        hvd.size(), hvd.local_size()))
    logging.info("Global rank: {}, local rank: {}".format(
        hvd.rank(), hvd.local_rank()))

    #strategy = tf.distribute.MirroredStrategy()
    # get the dataset
    #GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
    
    #num_workers = distribution_utils.configure_cluster(None, -1)
    #num_gpus = distribution_utils.get_num_gpus(8)
    #strategy = distribution_utils.get_distribution_strategy(
    #    distribution_strategy='multi_worker_mirrored',
    #    num_gpus=8,
    #    num_workers=32,
    #    all_reduce_alg="nccl")

    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
    #train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    #valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)
    #test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    #with strategy.scope():
    # create model
    model = get_model()
    print_model_summary(network=model)

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.RMSprop()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)


    #def compute_loss(logits, labels):
    #    per_example_loss = loss_object(y_true=labels, y_pred=logits)
    #    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    @tf.function
    def train_step(image_batch, label_batch, first_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)#compute_loss(logits=predictions, labels=label_batch)#
            
        tape = hvd.DistributedGradientTape(tape)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        if first_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)
        
    #@tf.function
    #def distributed_train_step(dist_inputs):
    #    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    #    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
    #                            axis=None)

    @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)#compute_loss(logits=predictions, labels=label_batch)#

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # start training
    for epoch in range(EPOCHS):
        step = 0
        for features in train_dataset:
            step += 1
            #images, labels = process_features(features, data_augmentation=True)
            images, labels = features
            train_step(images, labels, step == 0)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch,
                                                                                    EPOCHS,
                                                                                    step,
                                                                                    math.ceil(train_count / BATCH_SIZE),
                                                                                    train_loss.result().numpy(),
                                                                                    train_accuracy.result().numpy()))

        for features in valid_dataset:
            #valid_images, valid_labels = process_features(features, data_augmentation=False)
            valid_images, valid_labels, _, _ = features
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
            "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                EPOCHS,
                                                                train_loss.result().numpy(),
                                                                train_accuracy.result().numpy(),
                                                                valid_loss.result().numpy(),
                                                                valid_accuracy.result().numpy()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        if epoch % save_every_n_epoch == 0:
            #model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')
            if hvd.rank() == 0:
                checkpoint.save(save_model_dir)


    # save weights
    # model.save_weights(filepath=save_model_dir+"model", save_format='tf')

    # save the whole model
    # tf.saved_model.save(model, save_model_dir)

    # convert to tensorflow lite format
    # model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)

