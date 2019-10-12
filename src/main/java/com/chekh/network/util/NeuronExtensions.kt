package com.chekh.network.util

import com.chekh.network.neuron.FuzzyNeuron
import com.chekh.network.util.Functions.Companion.mul
import kotlin.math.ln
import kotlin.math.pow

fun FuzzyNeuron.derivativeC(
    ruleIndex: Int /*i*/,
    inputIndex: Int /*j*/,
    generatingIndex: Int /*k*/,
    x: Double,
    muList: List<List<Double>>
) = (Functions.kronecker(generatingIndex, ruleIndex) * mx(muList) - lx(muList[ruleIndex])) / mx(muList).pow(2) *
        muList[ruleIndex].mul(ignoreIndex = inputIndex) * 2 * b / sigma *
        ((x - c) / sigma).pow(2 * b - 1) / (1 + ((x - c) / sigma).pow(2 * b)).pow(2)

fun FuzzyNeuron.derivativeSigma(
    ruleIndex: Int,
    inputIndex: Int,
    generatingIndex: Int,
    x: Double,
    muList: List<List<Double>>
) = (Functions.kronecker(generatingIndex, ruleIndex) * mx(muList) - lx(muList[ruleIndex])) / mx(muList).pow(2) *
        muList[ruleIndex].mul(ignoreIndex = inputIndex) * 2 * b / sigma *
        ((x - c) / sigma).pow(2 * b) / (1 + ((x - c) / sigma).pow(2 * b)).pow(2)

fun FuzzyNeuron.derivativeB(
    ruleIndex: Int,
    inputIndex: Int,
    generatingIndex: Int,
    x: Double,
    muList: List<List<Double>>
) = (Functions.kronecker(generatingIndex, ruleIndex) * mx(muList) - lx(muList[ruleIndex])) / mx(muList).pow(2) *
        muList[ruleIndex].mul(ignoreIndex = inputIndex) * -2 * ((x - c) / sigma).pow(2 * b) *
        ln((x - c) / sigma) / (1 + ((x - c) / sigma).pow(2 * b)).pow(2)

private fun lx(muList: List<Double>) = muList.mul()

private fun mx(muList: List<List<Double>>) = muList.sumByDouble { it.mul() }