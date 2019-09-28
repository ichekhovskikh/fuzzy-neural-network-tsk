package com.chekh.network.layer

import com.chekh.network.neuron.SoftmaxNeuron

class SoftmaxLayer(val outputCount: Int) {
    var neurons: MutableList<SoftmaxNeuron> = mutableListOf()
        private set

    init {
        for (index in 0 until outputCount) {
            neurons.add(SoftmaxNeuron())
        }
    }

    fun calculate(summingLayer: SummingLayer) {
        require(outputCount == summingLayer.outputCount)
        neurons.forEachIndexed { index, neuron ->
            neuron.calculateOutput(summingLayer.functionalNeurons[index].sum, summingLayer.weightNeuron.sum)
        }
    }
}