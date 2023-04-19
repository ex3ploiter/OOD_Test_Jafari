import torch
from torchattacks import PGD as tPGD
from model import Model
from Attack import PGD,PGD_CLS,PGD_MSP
from utils import * 


attack_eps = 4/255
attack_steps = 10
attack_alpha = 2.5 * attack_eps / attack_steps


selected_model_adv='Robust_resnet18_linf_eps8.0'
pretrained='Pretrained'


def main(in_dataset,out_dataset):
    num_classes = {
        'CIFAR10': 10,
        'CIFAR100': 20,
        'MNIST': 10,
        'FashionMNIST': 10,
    }[in_dataset]

    model = Model(backbone=selected_model_adv, pretrained=pretrained, num_classes=num_classes).to(device)


    train_attack1 = PGD_CLS(model, eps=attack_eps, steps=10, alpha=attack_alpha)
    test_attack = PGD_MSP(model, eps=attack_eps, steps=10, alpha=attack_alpha, num_classes=num_classes)

    device = None

    try:
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    except:
        raise ValueError('Wrong CUDA Device!')

    trainloader,testloader=getLoaders(in_dataset,out_dataset)
    
    csv_file_name = f'{in_dataset}_vs_{out_dataset}_esp_{attack_eps}_steps_{attack_steps}_model_{selected_model_adv}.csv'
    clean_aucs, adv_aucs = run(csv_file_name, model, train_attack1, test_attack, trainloader, testloader, 1, 10, device, loss_threshold=1e-3, num_classes=num_classes)

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
    plt.show()


    dummy_attack_name= 'PGD-10'
    dummy_attack = PGD(model, eps=attack_eps, steps=10, alpha=attack_alpha, num_classes=num_classes)
    print(f'AUC & Accuracy Adversarial - {dummy_attack_name} - Started...')
    adv_auc, adv_accuracy = auc_softmax_adversarial(model=model, epoch=10, test_loader=testloader, test_attack=dummy_attack, device=device, num_classes=num_classes)
    print(f'AUC Adversairal {dummy_attack_name} - score on epoch {10} is: {adv_auc * 100}')
    print(f'Accuracy Adversairal {dummy_attack_name} -  score on epoch {10} is: {adv_accuracy * 100}')





import argparse


parser = argparse.ArgumentParser(description='Process input and output datasets.')
parser.add_argument('--in_dataset', type=str, help='Path to input dataset file.')
parser.add_argument('--out_dataset', type=str, help='Path to output dataset file.')
args = parser.parse_args()


main(args.in_dataset,args.out_dataset)

