# Copyright (c) 2019, Serge Guelton
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 	Redistributions of source code must retain the above copyright notice, this
# 	list of conditions and the following disclaimer.

# 	Redistributions in binary form must reproduce the above copyright notice,
# 	this list of conditions and the following disclaimer in the documentation
# 	and/or other materials provided with the distribution.

# 	Neither the name of HPCProject, Serge Guelton nor the names of its
# 	contributors may be used to endorse or promote products derived from this
# 	software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import parpy
import torch

@parpy.jit
def kernel(theta_1, phi_1, theta_2, phi_2, distance_matrix, N):
    parpy.label('i')
    for i in range(N):
        a = parpy.sin((theta_2[i] - theta_1[i]) / 2.0) ** 2.0
        b = parpy.cos(theta_1[i]) * parpy.cos(theta_2[i]) * \
            parpy.sin((phi_2[i] - phi_1[i]) / 2.0) ** 2.0
        temp = a + b
        distance_matrix[i] = 2.0 * (parpy.atan2(parpy.sqrt(temp), parpy.sqrt(1.0 - temp)))

def arc_distance(theta_1, phi_1, theta_2, phi_2, opts, compile_only=False):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    N, = theta_1.shape
    distance_matrix = torch.empty_like(theta_1)
    if compile_only:
        args = [theta_1, phi_1, theta_2, phi_2, distance_matrix, N]
        return parpy.print_compiled(kernel, args, opts)
    kernel(theta_1, phi_1, theta_2, phi_2, distance_matrix, N, opts=opts)
    return distance_matrix
