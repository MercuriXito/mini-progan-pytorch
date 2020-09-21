import torch
import torch.autograd as autograd

def calculate_gradient_penalty(netD, images, fake_images, gp_lambda, device, *args, **kws):

        batch_size, C, W, H = images.size()
        alpha = torch.randn((batch_size, 1), device=device)
        alpha = alpha.expand((batch_size, C * W * H)).contiguous()
        alpha = alpha.view_as(images)

        interpolate = alpha * images + (1 - alpha) * fake_images
        interpolate = interpolate.to(device)
        interpolate.requires_grad_(True)

        out = netD(interpolate, **kws)

        grads = autograd.grad(out, interpolate, 
        grad_outputs=torch.ones_like(out).type(torch.float).to(device), 
            retain_graph=True, create_graph=True)[0]
        grads = grads.view(grads.size(0), -1)

        return gp_lambda * ((grads.norm(p=2, dim = 1) - 1) ** 2).mean()

if __name__ == "__main__":
    pass
