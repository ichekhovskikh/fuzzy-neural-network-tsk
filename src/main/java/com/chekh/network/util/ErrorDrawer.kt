package com.chekh.network.util

interface ErrorDrawer {
    var errors: MutableList<Double>
    fun draw()
    fun clear()
}