package com.chekh.network.dataset

interface DatasetClassifier {
    fun getClass(index: Int, rows: List<Row>): List<Row>
    fun getClassesSize(rows: List<Row>): Int
}