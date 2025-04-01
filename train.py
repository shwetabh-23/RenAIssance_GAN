import torch
from train_utils import generator_loss, discriminator_loss, get_optimizers, compute_perceptual_loss, compute_canny_edge_loss, compute_l1_loss
from text_detection import batch_ssim_loss, compute_l1_loss_patches
from model import Generator, Discriminator
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
def train_step(input_image, target, generator, discriminator, 
               generator_loss, discriminator_loss, 
               generator_optimizer, discriminator_optimizer, 
               step, summary_writer):

    # Set models to training mode
    generator.train()
    discriminator.train()

    # Move data to GPU
    input_image, target = input_image.to("cuda"), target.to("cuda")

    # Forward pass through generator
    gen_output = generator(input_image)
    
    # Discriminator forward pass
    disc_real_output = discriminator(input_image, target)
    disc_generated_output = discriminator(input_image, gen_output.clone())  # Detach generator output to stop gradients
    # gen_clone = gen_output.clone()
    # Compute losses

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # Backpropagation for discriminator
    discriminator_optimizer.zero_grad()
    disc_loss.backward(retain_graph=True) 
    
    # Backpropagation for generator
    generator_optimizer.zero_grad()
    gen_total_loss.backward()
    discriminator_optimizer.step()
    generator_optimizer.step()
    
    # print('after backward', gen_total_loss.grad)  # Should have values now

    # Logging losses
    if step % 1000 == 0:
        with summary_writer:
            summary_writer.add_scalar('gen_total_loss', gen_total_loss.item(), step // 1000)
            summary_writer.add_scalar('gen_gan_loss', gen_gan_loss.item(), step // 1000)
            summary_writer.add_scalar('gen_l1_loss', gen_l1_loss.item(), step // 1000)
            summary_writer.add_scalar('disc_loss', disc_loss.item(), step // 1000)
    
    return gen_total_loss.item(), disc_loss.item()

def evaluate(input_image, target, generator, discriminator, generator_loss, discriminator_loss):
    # Set models to evaluation mode
    generator.eval()
    discriminator.eval()

    # Move data to GPU
    input_image, target = input_image.to("cuda"), target.to("cuda")

    with torch.no_grad():  # Disable gradient tracking
        # Forward pass through generator
        gen_output = generator(input_image)
        
        # Discriminator forward pass
        disc_real_output = discriminator(input_image, target)
        disc_generated_output = discriminator(input_image, gen_output)

        # Compute losses
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    return gen_total_loss.item(), disc_loss.item()

def train_step_new(original_image, texture, generator, 
               generator_optimizer, 
               step, summary_writer):

    # Set models to training mode
    generator.train()
    # discriminator.train()

    # Move data to GPU
    input_image, texture = original_image.to("cuda"), texture.to("cuda")
    # Forward pass through generator
    gen_output = generator(input_image)

    # Discriminator forward pass
    # disc_real_output = discriminator(input_image, target)
    # disc_generated_output = discriminator(input_image, gen_output.clone())  # Detach generator output to stop gradients
    # gen_clone = gen_output.clone()
    # Compute losses
    perceptual_loss = compute_perceptual_loss(gen_output, texture)
    l1_loss = F.l1_loss(gen_output, texture, reduction="mean")
    # canny_edge_loss = compute_canny_edge_loss(gen_output, input_image)
    # batch_ssim_loss_value = batch_ssim_loss(input_image, gen_output)
    compute_l1_loss_value = compute_l1_loss_patches(input_image, gen_output)
    # gen_total_loss = (0*perceptual_loss + canny_edge_loss)
    gen_total_loss = 100*compute_l1_loss_value + perceptual_loss + 10*l1_loss
    # gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    # disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # Backpropagation for discriminator
    # discriminator_optimizer.zero_grad()
    # disc_loss.backward(retain_graph=True) 
    # Backpropagation for generator
    generator_optimizer.zero_grad()
    gen_total_loss.backward()
    # discriminator_optimizer.step()
    generator_optimizer.step()

    # print('after backward', gen_total_loss.grad)  # Should have values now

    # Logging losses
    # if step % 1000 == 0:
    #     with summary_writer:
    #         summary_writer.add_scalar('gen_total_loss', gen_total_loss.item(), step // 1000)
            # summary_writer.add_scalar('perceptual_loss', perceptual_loss.item(), step // 1000)
            # summary_writer.add_scalar('canny_edge_loss', canny_edge_loss.item(), step // 1000)
            # summary_writer.add_scalar('disc_loss', disc_loss.item(), step // 1000)

    return gen_total_loss.item(), perceptual_loss.item(), compute_l1_loss_value.item(), l1_loss.item()  # perceptual_loss.item(), canny_edge_loss.item()

def evaluate_new(input_image, texture, generator):
    # Set models to evaluation mode
    generator.eval()
    # discriminator.eval()

    # Move data to GPU
    input_image, texture = input_image.to("cuda"), texture.to("cuda")

    with torch.no_grad():  # Disable gradient tracking
        # Forward pass through generator
        gen_output = generator(input_image)
        perceptual_loss = compute_perceptual_loss(gen_output, texture)
        # canny_edge_loss = compute_canny_edge_loss(gen_output, input_image)
        # batch_ssim_loss_value = batch_ssim_loss(gen_output, input_image)
        compute_l1_loss_value = compute_l1_loss_patches(input_image, gen_output)
        l1_loss = F.l1_loss(gen_output, texture, reduction="mean")
        # gen_total_loss = 0*perceptual_loss + canny_edge_loss
        gen_total_loss = 100*compute_l1_loss_value + perceptual_loss + 10*l1_loss
        # Discriminator forward pass
        # disc_real_output = discriminator(input_image, target)
        # disc_generated_output = discriminator(input_image, gen_output)

        # Compute losses
        # gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        # disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    return gen_total_loss.item(), perceptual_loss.item(), compute_l1_loss_value.item(), l1_loss.item() # , perceptual_loss.item(), canny_edge_loss.item()

if __name__ == '__main__' : 
    
    # Initialize models
    generator = Generator().to("cuda")
    discriminator = Discriminator().to("cuda")
    
    # Initialize optimizers
    generator_optimizer, discriminator_optimizer = get_optimizers(generator, discriminator)
    
    # Initialize dummy input
    dummy_input = torch.randn(16, 3, 256, 256).to("cuda")
    dummy_texture = torch.randn(16, 3, 256, 256).to("cuda")
    # Initialize SummaryWriter
    summary_writer = SummaryWriter()
    
    # Test train_step
    # gen_total_loss, disc_loss = train_step(dummy_input, dummy_input, generator, discriminator, 
    #                                        generator_loss, discriminator_loss, 
    #                                        generator_optimizer, discriminator_optimizer, 
    #                                        1000, summary_writer)
    
    gen_loss, disc_loss = train_step_new(dummy_input, dummy_texture, generator, 
               generator_optimizer, 
               1000, summary_writer)
    gen_total_loss, disc_loss = evaluate(dummy_input, dummy_input, generator, discriminator, generator_loss, discriminator_loss)
    print("Generator Total Loss:", gen_total_loss)
    print("Discriminator Loss:", disc_loss)
    
    print("train_step test passed")