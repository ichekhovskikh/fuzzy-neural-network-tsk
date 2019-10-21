package com.chekh.network.util

import com.chekh.network.dataset.Dataset

object DatasetSeparator {

    /**
     * first - train dataset
     * second - test dataset
     */
    fun separate(dataset: Dataset, trainPercent: Float): Pair<Dataset, Dataset> {
        val trainSize = (dataset.rows.size * trainPercent).toInt()
        if (trainSize == dataset.rows.size) {
            return dataset to Dataset(emptyList(), dataset.classifier)
        }
        val rows = dataset.rows.shuffled()
        val trainDataset = Dataset(rows.subList(0, trainSize), dataset.classifier)
        val testDataset = Dataset(rows.subList(trainSize, rows.size), dataset.classifier)
        return trainDataset to testDataset
    }
}