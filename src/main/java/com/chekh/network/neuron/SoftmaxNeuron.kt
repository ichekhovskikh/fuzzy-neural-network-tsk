package main.java.com.chekh.network.neuron

class SoftmaxNeuron {
    var y: Double = 0.0
        private set

    fun calculateOutput(signalFunction: Double, weightSum: Double): Double {
        y = signalFunction / weightSum
        return y
    }
}