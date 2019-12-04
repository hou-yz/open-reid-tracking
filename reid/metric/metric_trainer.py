import time
import torch


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class CNNTrainer(BaseTrainer):
    def __init__(self, model, criterion, ):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None, ):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(data_loader):
            feat1, feat2, target = data[0].cuda(), data[1].cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(feat1, feat2)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()

        for batch_idx, (data, target) in enumerate(test_loader):
            feat1, feat2, target = data[0].cuda(), data[1].cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(feat1, feat2)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()

        print('Test, Loss: {:.6f}, Prec: {:.1f}%, time: {:.1f}'.format(losses / (len(test_loader) + 1),
                                                                       100. * correct / (correct + miss),
                                                                       time.time() - t0))

        return losses / len(test_loader), correct / (correct + miss)
