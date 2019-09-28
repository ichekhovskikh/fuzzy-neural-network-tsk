package main.java.com.chekh.network.neuron

import main.java.com.chekh.network.util.Functions
import java.util.concurrent.ThreadLocalRandom

class FuzzyNeuron {
    var mu: Double = 0.0
        private set
    var b: Double = 0.0
        private set
    var c: Double = 0.0
        private set
    var sigma: Double = 0.0
        private set

    init {
        val random = ThreadLocalRandom.current()
        b = random.nextDouble(1.0)
        c = random.nextDouble(1.0)
        sigma = random.nextDouble(1.0)
    }

    fun calculateMu(x: Double): Double {
        mu = Functions.gaussian(x, b, c, sigma)
        return mu
    }
}