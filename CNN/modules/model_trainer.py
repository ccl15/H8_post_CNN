import tensorflow as tf
from collections import defaultdict


def evaluate_loss(model, dataset, loss_func):
    avg_loss = tf.keras.metrics.Mean(dtype=tf.float32)
    for image, attr, label in dataset:
        pred = model(image, attr, training=False)
        loss = loss_func(label, pred)
        avg_loss.update_state(loss)
    return avg_loss.result()


def train_model(
    model,
    datasets,
    summary_writer,
    saving_path,
    evaluate_freq,
    max_epoch,
    loss_name,
    L_rate,
    overfit_stop=None
):
    if loss_name == 'MSE':
        loss_function = tf.keras.losses.MeanSquaredError()
    elif loss_name == 'MAE':
        loss_function = tf.keras.losses.MeanAbsoluteError()
    
        
    optimizer = tf.keras.optimizers.Adam(float(L_rate) ) 
    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))

    @tf.function
    def train_step(image, attr, label, training=True):
        with tf.GradientTape() as tape:
            predict = model(image, attr, training=training)
            pred_loss = loss_function(predict, label)
        gradients = tape.gradient(pred_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        avg_losses[f'{loss_name}'].update_state(pred_loss)
        return
    
    best_loss = 9e10
    #start_save = 30
    print('Start training.')
    for epoch_index in range(1, max_epoch+1):
        # ---- train
        for image, attr, label in datasets['train']:
            train_step(image, attr, label, training=True)

        for loss_name, avg_loss in avg_losses.items():
            with summary_writer.as_default():
                tf.summary.scalar(loss_name, avg_loss.result(), step=epoch_index)
            avg_loss.reset_states()

        # ---- evaluate
        if (epoch_index % evaluate_freq) == 0:
        #if (epoch_index >= start_save) and (epoch_index % evaluate_freq == 0):
            print(f'Evaluate epochs {epoch_index}')
            
            for phase in ['valid']:
                loss  = evaluate_loss(model, datasets[phase], loss_function)
                with summary_writer.as_default():
                    tf.summary.scalar(f'[{phase}]: {loss_name}', loss, step=epoch_index) 
             
            valid_loss = loss
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch_index
                print(f'Get best loss. Save epoch {epoch_index}.')
                model.save_weights(f'{saving_path}/M', save_format='tf')
            elif overfit_stop and (epoch_index - best_epoch) >= overfit_stop:
                print('overfiting early stop!')
                break
            
