package com.chekh.network.dataset

class ClassificationDatasetClassifier : DatasetClassifier {

    override fun getClass(index: Int, rows: List<Row>): List<Row> {
        return rows.groupBy { it.output }.values.toList()[index]
    }

    override fun getClassesSize(rows: List<Row>): Int {
        return rows.groupBy { it.output.toInt() }.size
    }
}