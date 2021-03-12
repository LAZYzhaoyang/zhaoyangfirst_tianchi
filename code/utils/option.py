import torch
import numpy as np
import os

class option(object):
    def __init__(self):
        super(option, self).__init__()
        
        #==================data path==================#
        self.test_result_save_path = './prediction_result'
        self.save_model_path = './user_data/model_data'
        self.train_log = './user_data/train_log'
        
        self.train_path = './tcdata/suichang_round1_train_210120'
        #self.test_path = './tcdata/suichang_round1_test_partA_210120'
        self.test_path = './tcdata/suichang_round1_test_partB_210120'
        if not os.path.exists(self.train_path):
            self.train_path = '../tcdata/suichang_round1_train_210120'
            self.test_result_save_path = '../prediction_result'
            self.save_model_path = '../user_data/model_data'
            self.train_log = '../user_data/train_log'
        if not os.path.exists(self.test_path):
            #self.test_path = '../tcdata/suichang_round1_test_partA_210120'
            self.test_path = '../tcdata/suichang_round1_test_partB_210120'
            self.test_result_save_path = '../prediction_result'
            self.save_model_path = '../user_data/model_data'
            self.train_log = '../user_data/train_log'
        print('whether exists train path: ', os.path.exists(self.train_path))
        print('whether exist test path: ', os.path.exists(self.test_path))
        
        #==================model setting==================#
        self.last_activation = 'softmax'
        #self.last_activation = 'sigmoid'
        #self.last_activation = None
        
        #self.backbone = 'resnet50'
        #self.backbone = 'resnet34'
        self.backbone = 'efficientnet-b6'
        
        self.is_tta = False
        
        self.back_ground_index = None
        self.back_ground_threshold = None
        
        self.select_thresholds = [0.1, 1, 0.01, 0.05, 0.01, 0.05, 0.01, 0.01, 0.02, 0.01]
        
        #==================main model train setting==================#
        self.input_channel = 4
        self.n_classes = 10
        self.matches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.epochs = 150
        self.batch_size = 4
        self.lr = 0.001
        self.weight_decay = 5e-4
        self.momentum=0.8
        
        self.val_rate = 0.2
        self.shuffle_dataset = True
        
        #==================binary models setting==================#
        #self.class_weights = [1.5, 1, 3.6, 3.0, 2.3, 3.5, 2.0, 4.0, 1.7, 5.0]
        #self.class_weights = [1, 1, 1.3, 1, 1.3, 1.1, 1.25, 1.4, 1, 1.5]
        self.class_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        self.whether_train_model = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        #self.whether_train_model = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.Binary_class_index = [1, 3, 4, 5, 6, 7, 8, 9, 10]
        self.Binary_model_list = ['Unet', 'Unet++', 'PSPnet', 'PSPnet', 'Unet++', 'Linknet', 'Unet++','Unet', 'Unet++']
        self.Binary_backbone = ['resnet34', 'resnet50', 'resnet34', 'resnet34', 'resnet50', 'resnet34', 'resnet50', 'resnet34', 'resnet50']
        self.Binary_threshold = [0.1, 0.01, 0.02, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01]
        
        self.resize_rate = [0, 0.25]
        self.Binary_model_name = []
        for i in range(len(self.Binary_class_index)):
            name = 'Binary_model_of_class'+str(self.matches.index(self.Binary_class_index[i]))+'_'
            self.Binary_model_name.append(name)
            
        self.binary_lr = 0.0001
        self.binary_epochs = [10, 40, 60, 100, 30, 100, 60, 20, 30]
        self.binary_batch_size = 4
        self.binary_weight_decay = 5e-4
        self.binary_momentum=0.8
        
        #==================result save path==================#
        if not os.path.exists(self.test_result_save_path):
            os.mkdir(self.test_result_save_path)

        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
            
        if not os.path.exists(self.train_log):
            os.makedirs(self.train_log)

        self.best_model_name = self.save_model_path +'/best_loss_model.pth'
        self.save_model_name = self.save_model_path +'/epoch_model.pth'
        self.best_iou_model = os.path.join(self.save_model_path, 'checkpoint-best.pth')
        #==================visual setting==================#
        self.is_pretrain = False
        filename = os.path.join(opt.save_model_path, 'main_model' + 'checkpoint-best.pth')
        if os.path.exists(filename):
            self.is_pretrain = True
        
        self.show_inter = 5
        self.save_inter = 10
        self.min_inter = 5
        
        self.binary_threshold = 0.5

        # ====================================================
        self.alpha = 0.25
      
if __name__ == "__main__":
    Opt = option()




