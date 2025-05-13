import torch
import numpy as np

import modulus
from dataclasses import dataclass


from weatherlearn.models.pangu.pangu import EarthAttention3D, UpSample, DownSample, EarthSpecificBlock, BasicLayer
from weatherlearn.models.pangu.utils.shift_window_mask import get_shift_window_mask
from weatherlearn.models import Pangu, Pangu_lite, PanguPlasim
from weatherlearn.models.pangu.pangu_modulus import PanguPlasimModulus

import unittest


class TestMain(unittest.TestCase):
    def test_downsample(self):
        downsample_factor = 7
        in_dim = 1
        output_resolution = (8, 91, 180)
        input_resolution = (8, (output_resolution[1]-1)*downsample_factor+1,
                             output_resolution[2]*downsample_factor)

        x = torch.randn(1, np.prod(input_resolution), in_dim)
        downsample = DownSample(in_dim, input_resolution, output_resolution, downsample_factor=downsample_factor)
        x_downsample = downsample(x)
        self.assertEqual(x_downsample.shape, (1, 8 * 91 * 180, in_dim * downsample_factor))

    def test_upsample(self):
        upsample_factor = 7
        in_dim = upsample_factor
        out_dim = in_dim // upsample_factor
        input_resolution = (8, 91, 180)
        output_resolution = (8,
                             (input_resolution[1]-1)*upsample_factor+1,
                             input_resolution[2]*upsample_factor)
        upsample = UpSample(in_dim, out_dim, input_resolution, output_resolution, upsample_factor=upsample_factor)
        x = torch.randn(1, 8 * 91 * 180, in_dim)
        x_upsample = upsample(x)
        self.assertEqual(x_upsample.shape, (1, 8 * output_resolution[1] * output_resolution[2],
                                            x.shape[-1] // upsample_factor))

    def test_attention_without_mask1(self):
        input_resolution = (8, 186, 360)
        window_size = (2, 6, 12)
        num_heads = 2
        attention = EarthAttention3D(4, input_resolution, window_size, num_heads)
        batch_size = 2
        x = torch.randn(batch_size * 30, 4 * 31, 2 * 6 * 12, 4)
        attn = attention(x)
        self.assertEqual(attn.shape, x.shape)

    def test_attention_without_mask2(self):
        input_resolution = (8, 96, 180)
        window_size = (2, 6, 12)
        num_heads = 2
        attention = EarthAttention3D(4, input_resolution, window_size, num_heads)
        batch_size = 2
        x = torch.randn(batch_size * 15, 4 * 16, 2 * 6 * 12, 4)
        attn = attention(x)
        self.assertEqual(attn.shape, x.shape)

    def test_attention_with_mask(self):
        input_resolution = (8, 186, 360)
        window_size = (2, 6, 12)
        num_heads = 2
        attention = EarthAttention3D(4, input_resolution, window_size, num_heads)
        batch_size = 2
        x = torch.randn(batch_size * 30, 4 * 31, 2 * 6 * 12, 4)
        mask = get_shift_window_mask(input_resolution, window_size, (1, 3, 6))
        attn = attention(x, mask=mask)
        self.assertEqual(x.shape, attn.shape)

    def test_block_with_shift(self):
        dim = 4
        input_resolution = (8, 181, 360)
        num_heads = 2
        block = EarthSpecificBlock(dim, input_resolution, num_heads)
        batch_size = 1
        x = torch.randn(batch_size, 8 * 181 * 360, 4)
        block_x = block(x)
        self.assertEqual(x.shape, block_x.shape)

    def test_block_without_shift(self):
        dim = 4
        input_resolution = (8, 181, 360)
        num_heads = 2
        block = EarthSpecificBlock(dim, input_resolution, num_heads, shift_size=(0, 0, 0))
        batch_size = 1
        x = torch.randn(batch_size, 8 * 181 * 360, 4)
        block_x = block(x)
        self.assertEqual(x.shape, block_x.shape)

    def test_layer1(self):
        dim = 4
        input_resolution = (8, 181, 360)
        depth = 2
        num_heads = 2
        window_size = (2, 6, 12)
        layer = BasicLayer(dim, input_resolution, depth, num_heads, window_size)
        batch_size = 1
        x = torch.randn(batch_size, 8 * 181 * 360, dim)
        layer_x = layer(x)
        self.assertEqual(layer_x.shape, x.shape)

    def test_layer2(self):
        dim = 4
        input_resolution = (8, 91, 180)
        depth = 6
        num_heads = 2
        window_size = (2, 6, 12)
        layer = BasicLayer(dim, input_resolution, depth, num_heads, window_size)
        batch_size = 1
        x = torch.randn(batch_size, 8 * 91 * 180, dim)
        layer_x = layer(x)
        self.assertEqual(layer_x.shape, x.shape)

    """
    def test_pangu(self):
        pangu = Pangu()
        surface = torch.randn(1, 4, 721, 1440)
        surface_mask = torch.randn(3, 721, 1440)
        upper_air = torch.randn(1, 5, 13, 721, 1440)
        output_surface, output_upper_air = pangu(surface, surface_mask, upper_air)
        self.assertEqual(output_surface.shape, surface.shape)
        self.assertEqual(output_upper_air.shape, upper_air.shape)


    def test_compare_pangu_panguplasim(self):
        pangu = Pangu()
        pangu_plasim = PanguPlasim(horizontal_resolution = (721, 1440), num_levels = 13)
        surface = torch.randn(1, 4, 721, 1440)
        surface_mask = torch.randn(3, 721, 1440)
        upper_air = torch.randn(1, 5, 13, 721, 1440)
        output_surface, output_upper_air = pangu(surface, surface_mask, upper_air)
        output_surface_PP, output_upper_air_PP = pangu_plasim(surface, surface_mask, upper_air)
        self.assertEqual(output_surface.shape, surface.shape)
        self.assertEqual(output_upper_air.shape, upper_air.shape)
        self.assertEqual(output_surface_PP.shape, surface.shape)
        self.assertEqual(output_upper_air_PP.shape, upper_air.shape)
    """

    def test_panguplasim_modulus(self):
        pangu_plasim = PanguPlasim(horizontal_resolution=(65, 128), num_levels=10)
        pangu_plasim_modulus = PanguPlasimModulus(horizontal_resolution=(65, 128), num_levels=10)
        surface = torch.randn(1, 4, 65, 128)
        surface_mask = torch.randn(3, 65, 128)
        upper_air = torch.randn(1, 5, 10, 65, 128)
        output_surface_PP, output_upper_air_PP = pangu_plasim(surface, surface_mask, upper_air)
        output_surface_PPM, output_upper_air_PPM = pangu_plasim_modulus(surface, surface_mask, upper_air)
        self.assertEqual(output_surface_PP.shape, surface.shape)
        self.assertEqual(output_upper_air_PP.shape, upper_air.shape)
        print('PyTorch pangu_plasim okay!')
        self.assertEqual(output_surface_PPM.shape, surface.shape)
        self.assertEqual(output_upper_air_PPM.shape, upper_air.shape)
        print('Modulus pangu_plasim okay!')

    def test_panguplasim_modulus_cuda(self):
        if torch.cuda.is_available():
            pangu_plasim = PanguPlasim(horizontal_resolution=(65, 128), num_levels=10).cuda()
            pangu_plasim_modulus = PanguPlasimModulus(horizontal_resolution=(65, 128), num_levels=10).cuda()
            surface = torch.randn(1, 4, 65, 128).cuda()
            surface_mask = torch.randn(3, 65, 128).cuda()
            upper_air = torch.randn(1, 5, 10, 65, 128).cuda()
            output_surface_PP, output_upper_air_PP = pangu_plasim(surface, surface_mask, upper_air)
            output_surface_PPM, output_upper_air_PPM = pangu_plasim_modulus(surface, surface_mask, upper_air)
            self.assertEqual(output_surface_PP.shape, surface.shape)
            self.assertEqual(output_upper_air_PP.shape, upper_air.shape)
            print('PyTorch pangu_plasim cuda okay!')
            self.assertEqual(output_surface_PPM.shape, surface.shape)
            self.assertEqual(output_upper_air_PPM.shape, upper_air.shape)
            print('Modulus pangu_plasim cuda okay!')
        else:
            print('Skipping cuda test')

    def test_pangu_lite(self):
        pangu_lite = Pangu_lite(embed_dim=4, num_heads=(1, 1, 1, 1))
        surface = torch.randn(1, 4, 721, 1440)
        surface_mask = torch.randn(3, 721, 1440)
        upper_air = torch.randn(1, 5, 13, 721, 1440)
        output_surface, output_upper_air = pangu_lite(surface, surface_mask, upper_air)
        self.assertEqual(output_surface.shape, surface.shape)
        self.assertEqual(output_upper_air.shape, upper_air.shape)
