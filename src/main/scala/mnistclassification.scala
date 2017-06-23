/**
  * Created by abhishek on 9/6/17.
  */
import java.util

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.spark.api.TrainingMaster
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import java.util.ArrayList
import java.util.List

import org.apache.spark.rdd.RDD
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.SplitTestAndTrain


object mnistclassification {

  val nChannels = 1  // Number of input channels
  val outputNum = 10 // The number of possible outcomes
  val batchSize = 64 // Test batch size
  val nEpochs = 1    // Number of training epochs
  val iterations = 1 // Number of training iterations
  val seed = 123     //
  val learningRate=.01
  private val log: Logger = LoggerFactory.getLogger(mnistclassification.getClass)

    @throws[Exception]
    def main(args: Array[String]) {

      /*
         Create an iterator using the batch size for one iteration
       */

      log.info("Load data....")
      /*
      //for without spark
      val mnistTrain = new MnistDataSetIterator(batchSize, true, 100)
      val mnistTest = new MnistDataSetIterator(batchSize, false, 100)
      */

      // with spark RDD
      val examplesPerDataSetObject = 1
      val averagingFrequency = 3
      val batchSizePerWorker = 8
      //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
      val numLinesToSkip = 1
      val delimiter = ","
      val labelIndex=25
      val numClasses=3
      val quote = null
      /* val conf = new SparkConf().setMaster("local").setAppName("mnistclassification")
     //val sc: JavaSparkContext = new JavaSparkContext(conf)
     val sc: SparkContext = new SparkContext(conf)
   val sQLContext=new SQLContext(sc)

    val df=sQLContext.read.format("com.databricks.spark.csv").load("user_product.csv")
    val Array(trainData,testData) = df.randomSplit(Array(0.9, 0.1), seed = 12345)
    case class attributes(
                           Product_Id :String,
                                 Year : Int,
                                 Make : Int,
                           Model:String,
                           Submodel:String,
                           Engine:String,
                           Product_Type:String,
                           Category:String,
                           Sub_Category:String,
                           Position:Int,
                           Height:Int,
                           Weight:Int,
                           Price:Int,
                           Availability:Int,
                           User_Id:Int,
                           Viewed:Int,
                           Added:Int,
                           Removed:Int,
                           Purchased:Int,
                           Free_Shipping:Int,
                           Age:Int,
                           Gender:String,
                           Latitude:Int,
                           Longitude:Int,
                           Time_zone:Int,
                           User_Ratings:Int
        )
     val cs=df.as[attributes]
     println(cs)*/
      /*val recordReader: RecordReader  = new CSVRecordReader(numLinesToSkip,delimiter,quote)
      recordReader.initialize(new FileSplit(new ClassPathResource("user_product.csv").getFile()))

      val  iterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses)
      val DataList: List[DataSet] = new ArrayList[DataSet]()
      while (iterator.hasNext)
      {
         DataList.add(iterator.next)
      }
      val dd=sc.parallelize(DataList)
      val Array(trainData,testData) = dd.randomSplit(Array(0.9, 0.1), seed = 12345)*/

     val conf = new SparkConf().setMaster("local").setAppName("mnistclassification")

     val sc: JavaSparkContext = new JavaSparkContext(conf)

     val iterTrain: DataSetIterator = new MnistDataSetIterator(batchSizePerWorker, true, 12345)
     val iterTest: DataSetIterator =  new MnistDataSetIterator(batchSizePerWorker, true, 12345)

     val trainDataList: List[DataSet] = new ArrayList[DataSet]()
     val testDataList: List[DataSet] = new ArrayList[DataSet]()
     while (iterTrain.hasNext) {
      // val ds: DataSet = iterTrain.next()
       trainDataList.add(iterTrain.next)
     }

     while (iterTest.hasNext)
       {
         //val data: DataSet=iterTest.next()
         testDataList.add(iterTest.next)
       }

     val trainData = sc.parallelize(trainDataList)
     val testData = sc.parallelize(testDataList)

      //Set up the TrainingMaster. The TrainingMaster controls how learning is actually executed on Spark
        //Here, we are using standard parameter averaging
        //For details on these configuration options, see: https://deeplearning4j.org/spark#configuring

        val tm =new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
        .workerPrefetchNumBatches(2)    //Asynchronously prefetch up to 2 batches
        .averagingFrequency(averagingFrequency)
        .batchSizePerWorker(batchSizePerWorker)
        .build()


      /*
          Construct the neural network
       */

      log.info("Build model....")

      /*val conf=originalNetworkConfiguration
      val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
      model.init()*/

      //***************** Wide Model**********************************************************

      val linearconf=getRegressionLayerNetworkConfiguration
      val linearmodel: MultiLayerNetwork = new MultiLayerNetwork(linearconf)
      linearmodel.init()
      val linearW=linearmodel.getParam("0_W")

      //****************************Deep Model**********************************************

      val deepconf=getDeepLayerNetworkConfiguration
      val deepmodel: MultiLayerNetwork=new MultiLayerNetwork(deepconf)
      deepmodel.init()
      val deepW=deepmodel.getParam("0_W")
      println(deepmodel.paramTable().keySet())
      println(linearmodel.paramTable().keySet())
      val combinedW= deepW.addi(linearW)

     //***********************wide&Deep Model**************************************************

      val widedeepconf=getWideDeepLayerNetworkConfiguration
      val widedeepmodel: MultiLayerNetwork= new MultiLayerNetwork(widedeepconf)
      widedeepmodel.init()
      widedeepmodel.setParam("0_W",combinedW)

      //********* use Spark******************

      val sparkNetwork: SparkDl4jMultiLayer  = new SparkDl4jMultiLayer(sc, widedeepconf, tm)

      sparkNetwork.setListeners(new ScoreIterationListener(1))

      //****************************************************//
      log.info("Train model....")
      //widedeepmodel.setListeners(new ScoreIterationListener(1))
      for (i <- 0 until nEpochs) {
        sparkNetwork.fit(trainData)

        log.info("*** Completed epoch {} ***", i)
        log.info("Evaluate model....")
        val eval: Evaluation = new Evaluation(outputNum)

        }
     // val result=sparkNetwork.predict(mnistTest)
      val evaluation: Evaluation = sparkNetwork.evaluate(testData)
      log.info("***** Evaluation *****")
      log.info(evaluation.stats())
      //Delete the temp training files, now that we are done with them
      tm.deleteTempFiles(sc)

      }
      log.info("****************Example finished********************")


  private def getRegressionLayerNetworkConfiguration: MultiLayerConfiguration  = {

    new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .regularization(true).l1(0.01)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER).updater(Updater.NESTEROVS)
      .momentum(0.9)
      .list
      .layer(0, new ConvolutionLayer.Builder()
        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
        .nIn(nChannels)
        .stride(1, 1)
        .nOut(20)
        .activation(Activation.RELU)
        .build)
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY).nIn(nChannels).nOut(outputNum)
        .build)
      .pretrain(false).backprop(true)
      .build
  }
  private def getDeepLayerNetworkConfiguration: MultiLayerConfiguration  = {

    new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .regularization(true).l2(1e-6)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER).updater(Updater.ADAGRAD)
      .momentum(0.9)

      .list
      .layer(0, new ConvolutionLayer.Builder(5, 5)
        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
        .nIn(nChannels)
        .stride(1, 1)
        .nOut(20)
        .activation(Activation.RELU)
        .build)
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build)
      .layer(2, new ConvolutionLayer.Builder(5, 5)
        //Note that nIn need not be specified in later layers
        .stride(1, 1)
        .nOut(50)
        .activation(Activation.RELU)
        .build)
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build)
      .layer(4, new DenseLayer.Builder()
        .activation(Activation.RELU)
        .nOut(500)
        .build)
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .activation(Activation.IDENTITY).build)
      .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
      .backprop(true).pretrain(false).build

    /*
      Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
      (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
      and the dense layer
      (b) Does some additional configuration validation
      (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
      layer based on the size of the previous layer (but it won't override values manually set by the user)

      InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
      For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
      MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
      row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
      */

  }

  private def getWideDeepLayerNetworkConfiguration: MultiLayerConfiguration  = {

    new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .regularization(true).l2(0.0005)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER).updater(Updater.ADAM)
      .momentum(0.9)

      .list
      .layer(0, new ConvolutionLayer.Builder(5, 5)
        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
        .nIn(nChannels)
        .stride(1, 1)
        .nOut(20)
        .activation(Activation.SIGMOID)
        .build)
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build)
      .layer(2, new ConvolutionLayer.Builder(5, 5)
        //Note that nIn need not be specified in later layers
        .stride(1, 1)
        .nOut(50)
        .activation(Activation.SIGMOID)
        .build)
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build)
      .layer(4, new DenseLayer.Builder()
        .activation(Activation.SIGMOID)
        .nOut(500)
        .build)
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .activation(Activation.SOFTMAX).build)
      .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
      .backprop(true).pretrain(false).build
  }

  private def originalNetworkConfiguration: MultiLayerConfiguration  = {

    new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .regularization(true) // Training iterations as above
      .l2(0.0005)

      /*
        Uncomment the following for learning decay and bias
      */

      .learningRate(learningRate)//.biasLearningRate(0.02)

      //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)

      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .list
      .layer(0, new ConvolutionLayer.Builder(5, 5)
         //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
        .nIn(nChannels)
        .stride(1, 1)
        .nOut(20)
        .activation(Activation.IDENTITY)
        .build)
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build)
      .layer(2, new ConvolutionLayer.Builder(5, 5)
        //Note that nIn need not be specified in later layers
        .stride(1, 1)
        .nOut(50)
        .activation(Activation.IDENTITY)
        .build)
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build)
      .layer(4, new DenseLayer.Builder()
        .activation(Activation.RELU)
        .nOut(500)
        .build)
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .activation(Activation.SOFTMAX).build)
      .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
      .backprop(true).pretrain(false).build
  }

}
