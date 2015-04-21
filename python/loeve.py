#!/usr/bin/env python
#
# Copyright 2010-2013 Communications Engineering Lab, KIT
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

from gnuradio import gr
from gnuradio import blocks
from gnuradio import fft
import specest_gendpss
import specest_swig

## Estimates PSD using Thomson's multitaper method
# @param[in] N: Length of the FFT
# @param[in] NW: Time Bandwidth Product usually is of value 2, 2.5, 3.0, 3.5, or 4
# @param[in] K: Numbers of Tapers to use. K should be smaller than 2*NW
# @param[in] weighting: Which type of weighting to use for the eigenspectra. Choices can be 'unity','eigenvalues' or adaptive
class loeve(gr.hier_block2):
    """ Computes Loeve Spcetral Coherences """
    def __init__(self, N=512 , NW=3 , K=5, fftshift=False, samp_rate = 1, rate = 10):
        gr.hier_block2.__init__(self, "loeve",
                gr.io_signature(2, 2, gr.sizeof_gr_complex),
                gr.io_signature(1, 1, gr.sizeof_float*N))
        self.check_parameters(N, NW, K)

        self.s2v1 = blocks.stream_to_vector(gr.sizeof_gr_complex, N)
        self.s2v2 = blocks.stream_to_vector(gr.sizeof_gr_complex, N)
        self.one_in_n1 = blocks.keep_one_in_n(gr.sizeof_gr_complex * N, max(1, int(samp_rate/N/rate)))
        self.one_in_n2 = blocks.keep_one_in_n(gr.sizeof_gr_complex * N, max(1, int(samp_rate/N/rate)))
        self.connect((self, 0), self.s2v1, self.one_in_n1)
        self.connect((self, 1), self.s2v2, self.one_in_n2)

        dpss = specest_gendpss.gendpss(N=N, NW=NW, K=K)
        self.mtm1 = [eigenspectrum(dpss.dpssarray[i], fftshift) for i in xrange(K)]
        self.mtm2 = [eigenspectrum(dpss.dpssarray[i], fftshift) for i in xrange(K)]
        
        self.multipliers = [blocks.multiply_vcc(N) for i in xrange(K)]

        self.sum = blocks.add_vcc(N)
        self.divide = blocks.multiply_const_vcc([1./K]*N)
        self.c2mag = blocks.complex_to_mag_squared(N)
        self.connect_loeve(K)
        self.connect(self.sum, self.divide, self.c2mag, self)

    def connect_loeve(self, K):
        """ Connects up all the eigenspectrum calculators. """
        for i in xrange(K):
            self.connect(self.one_in_n1, self.mtm1[i])
            self.connect(self.one_in_n2, self.mtm2[i])
            self.connect(self.mtm1[i], (self.multipliers[i], 0))
            self.connect(self.mtm2[i], (self.multipliers[i], 1))
            
            self.connect(self.multipliers[i], (self.sum, i))


    ## Checks the validity of parameters
    # @param[in] N: Length of the FFT
    # @param[in] NW: Time Bandwidth Product
    # @param[in] K: Numbers of Tapers to used
    def check_parameters(self, N, NW, K):
        """ Checks the validity of parameters. """
        if NW < 1: raise ValueError, 'NW must be greater than or equal to 1'
        if K < 2:  raise ValueError, 'K must be greater than or equal to 2'
        if (N % 1): raise TypeError, 'N has to be an integer'
        if N < 1:  raise ValueError, 'N has to be greater than 1'


## Computes the eigenspectra for the multitaper spectrum estimator:
# data ----> multiplication dpss ----> FFT ----> square ----> output eigenspectrum
# @param[in] dpss: the dpss used as a data taper
class eigenspectrum(gr.hier_block2):
    """ Computes the eigenspectra for the multitaper spectrum estimator:
    data --> multiplication dpss --> FFT --> mag-square --> output eigenspectrum """
    def __init__(self, dpss, fftshift=False):
        gr.hier_block2.__init__(self, "eigenspectrum",
                gr.io_signature(1, 1, gr.sizeof_gr_complex*len(dpss)),
                gr.io_signature(1, 1, gr.sizeof_gr_complex*len(dpss)))
        self.window = dpss
        self.fft = fft.fft_vcc(len(dpss), True, self.window, fftshift)
        self.connect(self, self.fft, self)

