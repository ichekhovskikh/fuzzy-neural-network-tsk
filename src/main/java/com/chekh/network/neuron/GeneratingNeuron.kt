package com.chekh.network.neuron

import java.util.concurrent.ThreadLocalRandom

class GeneratingNeuron(inputCount: Int) {
    var generating: Double = 0.0
        private set
    var p: MutableList<Double> = mutableListOf()
        private set

    init {
        val random = ThreadLocalRandom.current()
        var accumulator = 0.0
        for (index in 0 until inputCount) {
            val value = random.nextDouble(1 - accumulator)
            accumulator += value
            p.add(value)
        }
        p.add(1 - accumulator)
    }

    fun calculateGenerating(x: List<Double>, weight: Double): Double {
        require(p.size - 1 == x.size)
        var sum = 0.0
        x.forEachIndexed { index, value -> sum += p[index + 1] * value }
        generating = weight * (p[0] + sum)
        return generating
    }
}