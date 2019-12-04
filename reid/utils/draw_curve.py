import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


def draw_curve(path, x_epoch, train_loss, train_prec, test_x_epoch=None, test_loss=None, test_prec=None, ):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="prec")
    ax0.plot(x_epoch, train_loss, 'bo-', label='train: {:.3f}'.format(train_loss[-1]))
    ax1.plot(x_epoch, train_prec, 'bo-', label='train: {:.3f}'.format(train_prec[-1]))
    if test_x_epoch:
        if test_loss:
            ax0.plot(test_x_epoch, test_loss, 'ro-', label='test: {:.3f}'.format(test_loss[-1]))
        if test_prec:
            ax1.plot(test_x_epoch, test_prec, 'ro-', label='test: {:.3f}'.format(test_prec[-1]))
    else:
        if test_loss:
            ax0.plot(x_epoch, test_loss, 'ro-', label='test: {:.3f}'.format(test_loss[-1]))
        if test_prec:
            ax1.plot(x_epoch, test_prec, 'ro-', label='test: {:.3f}'.format(test_prec[-1]))
    ax0.legend()
    ax1.legend()
    fig.savefig(path)
    plt.close(fig)
