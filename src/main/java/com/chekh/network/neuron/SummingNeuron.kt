package main.java.com.chekh.network.neuron

class SummingNeuron {
    var sum: Double = 0.0
        private set

    fun calculateSum(values: List<Double>): Double {
        sum = values.sum()
        return sum
    }
}