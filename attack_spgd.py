import torch
import torch.nn as nn

def cal_loss(loader,model,delta,beta,loss_function):
    loss_total = 0
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    delta = delta.cuda()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            x_val = data.cuda()
            outputs_ori = model(x_val.cuda())
            _, target_label = torch.max(outputs_ori, 1)
            perturbed = torch.clamp((x_val + delta), 0, 1)
            outputs = model(perturbed)
            if loss_function:
                loss = torch.mean(loss_fn(outputs, target_label))
            else:
                loss = torch.mean(outputs.gather(1, (target_label.cuda()).unsqueeze(1)).squeeze(1))
            loss_total = loss_total + loss
    loss_total = loss_total/(i+1)
    return loss_total


def uap_spgd(model, loader, nb_epoch, eps, beta=9, step_decay=0.1, loss_function=None,
            uap_init=None,batch_size = None,loader_eval = None, dir_uap = None,center_crop=224, Momentum=0, img_num=10000):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    step_decay  single step size
    loss_fn     custom loss function (default is CrossEntropyLoss)
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})
    center_crop image size
    Momentum    momentum item (default is false)
    
    log output
    batch_size  batch size 
    loader_eval evaluation dataloader
    dir_uap     save patch
    img_num     total image num
    ''' 
    model.eval()
    DEVICE = torch.device("cuda:0")
    if uap_init is None:
        batch_delta = torch.zeros(batch_size,3,center_crop,center_crop)  # initialize as zero vector
    else:
        batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    delta = batch_delta[0]
    losses = []
    
    # loss function
    if loss_function:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        beta = torch.cuda.FloatTensor([beta])

        def clamped_loss(output, target):
            loss = torch.mean(torch.min(loss_fn(output, target), beta))
            return loss


    batch_delta.requires_grad_()
    v = 0
    for epoch in range(nb_epoch):
        print('epoch %i/%i' % (epoch + 1, nb_epoch))

        # perturbation step size with decay
        eps_step = eps * step_decay

        for i, data in enumerate(loader):
            x_val = data
            with torch.no_grad():
                outputs_ori = model(x_val.cuda())
                _, target_label = torch.max(outputs_ori, 1)
            if i > 0 or epoch >0:
                batch_delta.grad.data.zero_()

            batch_delta.data = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])

            perturbed = torch.clamp((x_val + batch_delta).cuda(), 0, 1)
            outputs = model(perturbed)
            # loss function value
            if loss_function:
                loss = clamped_loss(outputs, target_label.cuda())
            else:
                loss = -torch.mean(outputs.gather(1, (target_label.cuda()).unsqueeze(1)).squeeze(1))
            loss.backward()
            # batch update
            #momentum
            if Momentum:
                batch_delta_grad = batch_delta.grad.data.mean(dim=0)
                if torch.norm(batch_delta_grad,p=1) == 0:
                    batch_delta_grad = batch_delta_grad
                else:
                    batch_delta_grad = batch_delta_grad / torch.norm(batch_delta_grad, p=1)
                v = 0.9*v + batch_delta_grad
                grad_sign = v.sign()
            else:
                grad_sign = batch_delta.grad.data.mean(dim=0).sign()

            delta = delta + grad_sign * eps_step
            delta = torch.clamp(delta, -eps, eps)
            batch_delta.grad.data.zero_()

        loss = cal_loss(loader_eval, model, delta.data, beta,loss_function)
        losses.append(torch.mean(loss.data).cpu())
        if (epoch+1) % 10 == 0:
            torch.save(delta.data,
                       dir_uap + 'spgd_' + '%d_%depoch_%dbatch.pth' % (img_num, epoch + 1, batch_size))


    return delta.data,losses