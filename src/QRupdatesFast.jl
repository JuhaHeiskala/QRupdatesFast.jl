# MIT License
# 
# Copyright (c) 2023 Juha Tapio Heiskala
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

module QRupdatesFast

import QRupdate_ng_jll
import LinearAlgebra

const BlasInt = LinearAlgebra.BlasInt

"""
    qrshift(Q::AbstractArray, R::AbstractArray, I::Int, J::Int)
    
Perform circular shift of QR factorization in the range given `I` and `J`.
"""
function qrshift(Q::AbstractArray, R::AbstractArray, I::Int, J::Int)
    cpQ = copy(Q)
    cpR = copy(R)
    qrshift!(cpQ, cpR, I, J)
    return (cpQ, cpR)
end
 
"""
    qrshift!(Q::AbstractArray, R::AbstractArray, I::Int, J::Int)
    
Perform inplace circular shift of QR factorization in the range given `I` and `J`.
The input matrices `Q` and `R` are overwritten.
"""
function qrshift!(Q::AbstractArray, R::AbstractArray, I::Int, J::Int)    
    qrshc!(Q, R, I, J)
end


# The below interface description copyright and license as below:
#
# Copyright (C) 2008, 2009  VZLU Prague, a.s., Czech Republic
#
# Author: Jaroslav Hajek <highegg@gmail.com>
#
# This file is part of qrupdate.
#
# qrupdate is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, see
# <http://www.gnu.org/licenses/>.
#
# purpose:      updates a QR factorization after circular shift of
#               columns.
#               i.e., given an m-by-k orthogonal matrix Q, an k-by-n
#               upper trapezoidal matrix R and index j in the range
#               1:n+1, this subroutine updates the matrix Q -> Q1 and
#               R -> R1 so that Q1 is again orthogonal, R1 upper
#               trapezoidal, and
#               Q1*R1 = A(:,p), where A = Q*R and p is the permutation
#               [1:i-1,shift(i:j,-1),j+1:n] if i < j  or
#               [1:j-1,shift(j:i,+1),i+1:n] if j < i.
#               (real version)
# arguments:
# m (in)        number of rows of the matrix Q.
# n (in)        number of columns of the matrix R.
# k (in)        number of columns of Q1, and rows of R1. Must be
#               either k = m (full Q) or k = n <= m (economical form).
# Q (io)        on entry, the unitary m-by-k matrix Q.
#               on exit, the updated matrix Q1.
# ldq (in)      leading dimension of Q. ldq >= m.
# R (io)        on entry, the original matrix R.
#               on exit, the updated matrix R1.
# ldr (in)      leading dimension of R. ldr >= k.
# i (in)        the first index determining the range (see above)
# j (in)        the second index determining the range (see above)
# w (o)         a workspace vector of size 2*k.
for (f, elty) in ((:(:sqrshc_), :Float32), (:(:dqrshc_), :Float64))
    @eval begin
        function qrshc!(Q::AbstractMatrix{$elty}, R::AbstractMatrix{$elty},
                        i::Int, j::Int)

            Base.require_one_based_indexing(Q, R)
            LinearAlgebra.chkstride1(Q)
            LinearAlgebra.chkstride1(R)

            m = Int32(size(Q, 1))
            n = Int32(size(R, 2))
            k = Int32(size(R, 1))
            ldq = size(Q, 1)
            ldr = size(R, 1)
            
            work = Vector{$elty}(undef, 2*k)

            ccall(($f, QRupdate_ng_jll.libqrupdate), Cvoid,
                  (#    M             N            K
                   Ref{Int32}, Ref{Int32}, Ref{Int32},
                   #    Q           LDQ           R 
                   Ptr{$elty}, Ref{Int32}, Ptr{$elty},                       
                   #    LDR           I           J
                   Ref{Int32}, Ref{Int32}, Ref{Int32},
                   #   work  
                   Ptr{$elty}),
                  m, n, k, Q, ldq, R, ldr, i, j, work)

            return ()
            
        end
    end
end

for (f, elty,rty) in ((:(:cqrshc_), :ComplexF32, :Float32),
                      (:(:zqrshc_), :ComplexF64, :Float64))
    @eval begin
        function qrshc!(Q::AbstractMatrix{$elty}, R::AbstractMatrix{$elty},
                        i::Int, j::Int)

            Base.require_one_based_indexing(Q, R)
            LinearAlgebra.chkstride1(Q)
            LinearAlgebra.chkstride1(R)

            m = Int32(size(Q, 1))
            n = Int32(size(R, 2))
            k = Int32(size(R, 1))
            ldq = size(Q, 1)
            ldr = size(R, 1)
            
            work = Vector{$elty}(undef, k)
            rwork = Vector{$rty}(undef, k)
            ccall(($f, QRupdate_ng_jll.libqrupdate), Cvoid,
                  (#    M             N            K
                   Ref{Int32}, Ref{Int32}, Ref{Int32},
                   #    Q           LDQ           R 
                   Ptr{$elty}, Ref{Int32}, Ptr{$elty},                       
                   #    LDR           I           J
                   Ref{Int32}, Ref{Int32}, Ref{Int32},
                   #   work    real work
                   Ptr{$elty}, Ptr{$rty}),
                    m, n, k, Q, ldq, R, ldr, i, j, work, rwork)

            return ()
            
        end
    end
end


# purpose:      updates an LU factorization after rank-1 modification
#               i.e., given an m-by-k lower-triangular matrix L with unit
#               diagonal and a k-by-n upper-trapezoidal matrix R,
#               where k = min(m,n),
#               this subroutine updates L -> L1 and R -> R1 so that
#               L is again lower unit triangular, R upper trapezoidal,
#               and L1*R1 = L*R + u*v.'.
#               (real version)
# arguments:
# m (in)        order of the matrix L.
# n (in)        number of columns of the matrix U.
# L (io)        on entry, the unit lower triangular matrix L.
#               on exit, the updated matrix L1.
# ldl (in)      the leading dimension of L. ldl >= m.
# R (io)        on entry, the upper trapezoidal m-by-n matrix R.
#               on exit, the updated matrix R1.
# ldr (in)      the leading dimension of R. ldr >= min(m,n).
# u (io)        the left m-vector. On exit, if k < m, u is destroyed.
# v (io)        the right n-vector. On exit, v is destroyed.
#
# REMARK:       Algorithm is due to
#               J. Bennett: Triangular factors of modified matrices,
#                           Numerische Mathematik, 7 (1965)
#
for (f, elty) in ((:(:slu1up_), :Float32), (:(:dlu1up_), :Float64),
                  (:(:clu1up_), :ComplexF32), (:(:zlu1up_), :ComplexF64))
    @eval begin
        function lu1up!(L::AbstractMatrix{$elty}, R::AbstractMatrix{$elty},
                        x::AbstractVector{$elty}, y::AbstractVector{$elty})

            Base.require_one_based_indexing(L, R)
            LinearAlgebra.chkstride1(L)
            LinearAlgebra.chkstride1(R)

            m = Int32(size(L, 1))
            n = Int32(size(R, 2))
            ldl = size(L, 1)
            ldr = size(R, 1)
            
           
            ccall(($f, QRupdate_ng_jll.libqrupdate), Cvoid, 
                  (#    M             N            L
                   Ref{Int32}, Ref{Int32}, Ptr{$elty},
                   #    LDL           R           LDR 
                   Ref{Int32}, Ptr{$elty}, Ref{Int32},                       
                   #    U             V
                   Ptr{$elty}, Ptr{$elty}),
                  m, n, L, ldl, R, ldr, x, y)

            return ()
            
        end
    end
end

#      subroutine (s,d,c,z)lup1up(m,n,L,ldl,R,ldr,p,u,v,w)
# purpose:      updates a row-pivoted LU factorization after rank-1 modification
#               i.e., given an m-by-k lower-triangular matrix L with unit
#               diagonal, a k-by-n upper-trapezoidal matrix R, and a
#               permutation matrix P, where k = min(m,n),
#               this subroutine updates L -> L1, R -> R1 and P -> P1 so that
#               L is again lower unit triangular, R upper trapezoidal,
#               P permutation and P1'*L1*R1 = P'*L*R + u*v.'.
#               (real version)
# arguments:
# m (in)        order of the matrix L.
# n (in)        number of columns of the matrix U.
# L (io)        on entry, the unit lower triangular matrix L.
#               on exit, the updated matrix L1.
# ldl (in)      the leading dimension of L. ldl >= m.
# R (io)        on entry, the upper trapezoidal m-by-n matrix R.
#               on exit, the updated matrix R1.
# ldr (in)      the leading dimension of R. ldr >= min(m,n).
# p (in)        the permutation vector representing P
# u (in)        the left m-vector.
# v (in)        the right n-vector.
# w (work)      a workspace vector of size m.
#
# REMARK:       Algorithm is due to
#               A. Kielbasinski, H. Schwetlick, Numerische Lineare
#               Algebra, Verlag Harri Deutsch, 1988
#
for (f, elty) in ((:(:slup1up_), :Float32), (:(:dlup1up_), :Float64),
                  (:(:clup1up_), :ComplexF32), (:(:zlup1up_), :ComplexF64))
    @eval begin
        function lup1up!(L::AbstractMatrix{$elty}, R::AbstractMatrix{$elty},
                         p::AbstractVector{Int32},
                         u::AbstractVector{$elty}, v::AbstractVector{$elty})

            Base.require_one_based_indexing(L, R)
            LinearAlgebra.chkstride1(L)
            LinearAlgebra.chkstride1(R)

            m = Int32(size(L, 1))
            n = Int32(size(R, 2))
            ldl = size(L, 1)
            ldr = size(R, 1)
            
            work = Vector{$elty}(undef, m)

            ccall(($f, QRupdate_ng_jll.libqrupdate), Cvoid,
                  (#    M             N            L
                   Ref{Int32}, Ref{Int32}, Ptr{$elty},
                   #    LDL           R           LDR 
                   Ref{Int32}, Ptr{$elty}, Ref{Int32},                       
                   #    P           U           V
                   Ptr{Int32}, Ptr{$elty}, Ptr{$elty},
                   #   WORK
                   Ptr{$elty}),
                  m, n, L, ldl, R, ldr, p, u, v, work)

            return ()
            
        end
    end
end

end # module

