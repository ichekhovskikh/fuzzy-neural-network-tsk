package com.chekh.network.util

import org.ejml.simple.SimpleMatrix

fun List<Double>.toSimpleColumn(): SimpleMatrix {
    val array: Array<DoubleArray> = Array(size) { index -> doubleArrayOf(this[index]) }
    return SimpleMatrix(array)
}

fun List<DoubleArray>.toSimpleMatrix(): SimpleMatrix {
    return SimpleMatrix(toTypedArray())
}

fun SimpleMatrix.toList(column: Int): List<Double> {
    val list = mutableListOf<Double>()
    for (index in 0 until numRows()) {
        list.add(this[index, column])
    }
    return list
}