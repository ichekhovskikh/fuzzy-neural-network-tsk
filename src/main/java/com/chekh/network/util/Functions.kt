package main.java.com.chekh.network.util

import kotlin.math.pow

class Functions private constructor() {

    companion object {
        @JvmStatic
        fun gaussian(x: Double, b: Double, c: Double, sigma: Double): Double =
            1.0 / (1 + ((x - c) / sigma).pow(2 * b))

        @JvmStatic
        fun List<Double>.mul(): Double {
            var mul = 0.0
            this.forEach { mul *= it }
            return mul
        }
    }
}