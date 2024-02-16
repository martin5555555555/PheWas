import os
import sys
import time
import matplotlib.pyplot as plt
def clear_last_line(output_file):
    with open(output_file, 'r') as file:
        file.seek(0)
        lignes = file.readlines()
        file.close()
    lignes_new = lignes[:-1]
    

    with open(output_file, 'w') as file:
        file.writelines(lignes_new)
        file.close()



def print_file(filename, message, new_line=True):
    with open(filename, 'a') as file:
        if new_line:
            file.write('\n'+message)
        else:
            file.write(message)
        file.close()
def number_tests(model_dir):
    if os.listdir(model_dir) == []:
        return 1
    else:
        last_test_nb = max([ int(test_names.split('_')[0]) for test_names in os.listdir(model_dir)])
        return last_test_nb + 1

# Unable bufffering for standard out
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
   
def plot_ini_infos(model, output_file, dataloader_test, dataloader_train, writer, dic_features_list):
        f1_val, accuracy_val, auc_score_val, loss_val, proba_avg_zero_val, proba_avg_one_val, predicted_probas_list_val, true_labels_list_val = model.evaluate(dataloader_test)
        f1_train, accuracy_train, auc_score_train, loss_train, proba_avg_zero_train, proba_avg_one_train, predicted_probas_list_train, true_labels_list_train = model.evaluate(dataloader_train)

        print_file(output_file, " Evaluation on validation", new_line=True)
        print_file(output_file, f"        Validation loss : {loss_val:.4f}", new_line=True)
        print_file(output_file, f"    Training loss : {(loss_train/len(dataloader_train)):.4f}", new_line = True)

        
        writer.add_scalar('Validation loss', loss_val, 0)
        writer.add_scalar('Validation AUC', auc_score_val, 0)
        writer.add_scalar('Validation f1-score', f1_val, 0)
        writer.add_scalar('Validation accuracy', accuracy_val, 0)
        writer.add_scalar('Validation Average Proba 0', proba_avg_zero_val, 0)
        writer.add_scalar('Validation Average Proba 1', proba_avg_one_val, 0)


        writer.add_scalar('Training loss', loss_train, 0)
        writer.add_scalar('Training AUC', auc_score_train, 0)
        writer.add_scalar('Training f1-score', f1_train, 0)
        writer.add_scalar('Training accuracy', accuracy_train, 0)
        writer.add_scalar('Training Average Proba 0', proba_avg_zero_train, 0)
        writer.add_scalar('Training Average Proba 1', proba_avg_one_train, 0)


        print_file(output_file, 'will be deleted', new_line=True)

        dic_features_list['list_training_loss'].append(loss_train)
        dic_features_list['list_validation_loss'].append(loss_val)
        dic_features_list['list_proba_avg_zero'].append(proba_avg_zero_val)
        dic_features_list['list_proba_avg_one'].append(proba_avg_one_val)
        dic_features_list['list_auc_validation'].append(auc_score_val)
        dic_features_list['list_accuracy_validation'].append(accuracy_val)
        dic_features_list['list_f1_validation'].append(f1_val)
        dic_features_list['epochs'].append(0)


        dic_features_epoch = {
             'Validation loss' : loss_val,
             'Validation AUC' : auc_score_val,
             'Validation f1-score' : f1_val,
             'Validation accuracy' : accuracy_val,
             'Validation Average Proba 0' : proba_avg_zero_val,
             'Validation Average Proba 1' : proba_avg_one_val,

             'Training loss' : loss_train,
             'Training AUC' : auc_score_train,
             'Training f1-score' : f1_train,
             'Training accuracy' : accuracy_train,
             'Training Average Proba 0' : proba_avg_zero_train,
             'Training Average Proba 1' : proba_avg_one_train,


        }


        return dic_features_epoch


def plot_infos(model, output_file, epoch, total_loss, start_time_epoch, dataloader_train, dataloader_test, optimizer, writer, dic_features_list, plots_path):
        print_file(output_file, f"Epoch {epoch} finished: {int(time.time() - start_time_epoch)} seconds", new_line=True)
        print_file(output_file, f"    Training loss : {(total_loss/len(dataloader_train)):.4f}", new_line = True)
        f1_val, accuracy_val, auc_score_val, loss_val, proba_avg_zero_val, proba_avg_one_val, predicted_probas_list_val, true_labels_list_val = model.evaluate(dataloader_test)
        f1_train, accuracy_train, auc_score_train, loss_train, proba_avg_zero_train, proba_avg_one_train, predicted_probas_list_train, true_labels_list_train = model.evaluate(dataloader_train)

        print_file(output_file, " Evaluation on validation", new_line=True)
        print_file(output_file, f"        Validation loss : {loss_val:.4f}", new_line=True)


        writer.add_scalar('Validation loss', loss_val, epoch)
        writer.add_scalar('Validation AUC', auc_score_val, epoch)
        writer.add_scalar('Validation f1-score', f1_val, epoch)
        writer.add_scalar('Validation accuracy', accuracy_val, epoch)
        writer.add_scalar('Validation Average Proba 0', proba_avg_zero_val, epoch)
        writer.add_scalar('Validation Average Proba 1', proba_avg_one_val, epoch)


        writer.add_scalar('Training loss', loss_train, epoch)
        writer.add_scalar('Training AUC', auc_score_train, epoch)
        writer.add_scalar('Training f1-score', f1_train, epoch)
        writer.add_scalar('Training accuracy', accuracy_train, epoch)
        writer.add_scalar('Training Average Proba 0', proba_avg_zero_train, epoch)
        writer.add_scalar('Training Average Proba 1', proba_avg_one_train, epoch)



        print_file(output_file, f"learning rate : {optimizer.param_groups[0]['lr']}", new_line=True)
        print_file(output_file, 'will be deleted', new_line=True)

        dic_features_list['list_training_loss'].append(total_loss/len(dataloader_train))
        dic_features_list['list_validation_loss'].append(loss_val)
        dic_features_list['list_proba_avg_zero'].append(proba_avg_zero_val)
        dic_features_list['list_proba_avg_one'].append(proba_avg_one_val)
        dic_features_list['list_auc_validation'].append(auc_score_val)
        dic_features_list['list_accuracy_validation'].append(accuracy_val)
        dic_features_list['list_f1_validation'].append(f1_val)
        dic_features_list['epochs'].append(epoch)


        dic_features_epoch = {
             'Validation loss' : loss_val,
             'Validation AUC' : auc_score_val,
             'Validation f1-score' : f1_val,
             'Validation accuracy' : accuracy_val,
             'Validation Average Proba 0' : proba_avg_zero_val,
             'Validation Average Proba 1' : proba_avg_one_val,

             'Training loss' : loss_train,
             'Training AUC' : auc_score_train,
             'Training f1-score' : f1_train,
             'Training accuracy' : accuracy_train,
             'Training Average Proba 0' : proba_avg_zero_train,
             'Training Average Proba 1' : proba_avg_one_train,


        }

        plots_plots(plots_path, dic_features_list)

        return dic_features_epoch


def plots_plots(plots_path, dic_features):
    list_epochs = dic_features['epochs']
    nb_fig = len(list(dic_features.keys()))
    fig, axes = plt.subplots(nb_fig-1,1 , figsize=(10, 10))  # -1 because of epochs

    for k, key in enumerate(dic_features):
        if key != 'epochs':
            list_value = dic_features[key]
            axes[k].plot(list_epochs, list_value)
            axes[k].set_xlabel('epochs')
            axes[k].set_ylabel(key)
            axes[k].set_title(key)
    fig.tight_layout()
    plt.savefig(plots_path)

