package com.chekh.network.util

import kotlin.math.pow

class Functions private constructor() {

    companion object {
        @JvmStatic
        fun gaussian(x: Double, b: Double, c: Double, sigma: Double): Double =
            1.0 / (1 + ((x - c) / sigma).pow(2 * b))

        @JvmStatic
        @JvmOverloads
        fun List<Double>.mul(ignoreIndex: Int = -1): Double {
            var mul = 0.0
            this.forEachIndexed { index, value ->
                if (index != ignoreIndex) {
                    mul *= value
                }
            }
            return mul
        }

        @JvmStatic
        fun kronecker(i: Int, j: Int) = if (i == j) 1 else 0
    }
}