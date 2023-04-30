import torch
from torchattacks import PGD as tPGD
from model import Model,Model_Pretrain,Model_FromScratch
from Attack import PGD,PGD_CLS,PGD_MSP
from utils import * 
import os




device = None

try:
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
except:
    raise ValueError('Wrong CUDA Device!')

# attack_eps = 4/255
attack_steps = 10
# attack_alpha = 2.5 * attack_eps / attack_steps


# selected_model_adv='Robust_resnet18_linf_eps8.0'
# pretrained='Pretrained'


def main(in_dataset,out_dataset,batch_size,pretrain):
    # if pretrain==True:
    #     attack_eps = 4/255
    #     selected_model_adv="Pang2022Robustness_WRN28_10"
    
    # else:
    #     attack_eps = 8/255
    #     selected_model_adv="WideResNet"

    


    for folder in ['./Results/','./CheckPoints/','./Logs/','./Plots/']:
        if not os.path.exists(folder):
            os.makedirs(folder)   
    
    num_classes = {
        'CIFAR10': 10,
        'CIFAR100': 20,
        'MNIST': 10,
        'FashionMNIST': 10,
    }[in_dataset]

    # model = Model(backbone=selected_model_adv, pretrained=pretrained, num_classes=num_classes).to(device)

    if pretrain=='False': # From Scratch
        model=Model_FromScratch(num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5, 0.999))
        lr=0.1
        
        attack_eps = 8/255
        selected_model_adv="WideResNet"        

    
    else :
        model=Model_Pretrain(num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
        lr=0.01
        
        attack_eps = 4/255
        selected_model_adv="Pang2022Robustness_WRN28_10"        
    
    attack_alpha = 2.5 * attack_eps / attack_steps

    train_attack1 = PGD_CLS(model, eps=attack_eps, steps=10, alpha=attack_alpha)
    test_attack = PGD_MSP(model, eps=attack_eps, steps=10, alpha=attack_alpha, num_classes=num_classes)


    trainloader,testloader=getLoaders(in_dataset,out_dataset,batch_size)

    
    
    csv_file_name = f'./Results/{in_dataset}_vs_{out_dataset}_esp_{attack_eps}_steps_{attack_steps}_model_{selected_model_adv}.csv'
    

    clean_aucs, adv_aucs = run(csv_filename=csv_file_name,model= model, train_attack=train_attack1,test_attack= test_attack, trainloader=trainloader
                               , testloader=testloader, 
                               test_step=1, max_epochs=10, device=device, loss_threshold=1e-3, num_classes=num_classes,optimizer=optimizer,lr=lr)
    
    
    
    
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 7))
    plt.plot(
        clean_aucs, color='green', linestyle='-', 
        label='clean AUC'
    )
    plt.plot(
        adv_aucs, color='blue', linestyle='-', 
        label='adversarial auc'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()
    plt.savefig(f'./Plots/{in_dataset}_vs_{out_dataset}_esp_{attack_eps}_steps_{attack_steps}_model_{selected_model_adv}_Pretrain_{pretrain}.png')



    
    general_logger = GeneralLogger(f'./Logs/{in_dataset}_vs_{out_dataset}_esp_{attack_eps}_steps_{attack_steps}_model_{selected_model_adv}_Pretrain_{pretrain}.log')
    
    dummy_attack_name= 'PGD-40'
    # dummy_attack = PGD(model, eps=attack_eps, steps=10, alpha=attack_alpha, num_classes=num_classes)
    dummy_attack = PGD_MSP(model, eps=attack_eps, steps=40, alpha=attack_alpha, num_classes=num_classes)
    # print(f'AUC & Accuracy Adversarial - {dummy_attack_name} - Started...')
    general_logger.log(f'AUC & Accuracy Adversarial - {dummy_attack_name} - Started...')
    adv_auc, adv_accuracy = auc_softmax_adversarial(model=model, epoch=10, test_loader=testloader, test_attack=dummy_attack, device=device, num_classes=num_classes)
    # print(f'AUC Adversairal {dummy_attack_name} - score on epoch {10} is: {adv_auc * 100}')
    general_logger.log(f'AUC Adversairal {dummy_attack_name} - score on epoch {10} is: {adv_auc * 100}')
    # print(f'Accuracy Adversairal {dummy_attack_name} -  score on epoch {10} is: {adv_accuracy * 100}')
    general_logger.log(f'Accuracy Adversairal {dummy_attack_name} -  score on epoch {10} is: {adv_accuracy * 100}')
    
    
    
    
    checkpoint_path=f'./CheckPoints/{in_dataset}_vs_{out_dataset}_esp_{attack_eps}_steps_{attack_steps}_model_{selected_model_adv}_Pretrain_{pretrain}.cpkt' 
    torch.save(model.state_dict(), checkpoint_path)


import argparse


parser = argparse.ArgumentParser(description='Process input and output datasets.')
parser.add_argument('--in_dataset', type=str, help='Path to input dataset file.')
parser.add_argument('--out_dataset', type=str, help='Path to output dataset file.')
parser.add_argument('--batch_size',default=128, type=int, help='Path to output dataset file.')
parser.add_argument('--pretrain',default='False', type=str)

args = parser.parse_args()


main(args.in_dataset,args.out_dataset,args.batch_size,args.pretrain)

