package com.chekh.network

data class Dataset(var rows: List<Row>)

data class Row(var inputs: List<Double>, var output: Double)