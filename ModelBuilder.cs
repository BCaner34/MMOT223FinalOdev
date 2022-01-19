// ******************************************************* **********************************************
// * *
// * Bu, Microsoft ML.NET CLI (Komut Satırı Arayüzü) aracı tarafından otomatik olarak oluşturulan bir dosyadır. *
// * *
// ******************************************************* **********************************************

 Sistem kullanarak ;
 Sistemi kullanarak . Koleksiyonlar _ genel ;
 Sistemi kullanarak . IO ;
 Sistemi kullanarak . Linq ;
 Microsoft'u kullanarak . makine öğrenimi ;
 Microsoft'u kullanarak . ML . Veri ;
 PricePredictionML kullanarak . modeli . Veri Modelleri ;

ad alanı  PricePredictionML . Konsol Uygulaması
{
    genel  statik  sınıf  ModelBuilder
{
    özel  statik  dize  TRAIN_DATA_FILEPATH  =  @" G:\C# Köşe Öğeleri\Fiyat Tahmini\price-data.csv " ;
    özel  statik  dize  MODEL_FILEPATH  =  @" ../../../../Price PredictionML.Model/MLModel.zip " ;

    // Model oluşturma iş akışı nesneleri arasında paylaşılacak MLContext oluşturun
    // Birden çok eğitimde tekrarlanabilir/belirleyici sonuçlar için rastgele bir tohum ayarlayın.
    özel  statik  MLContext  mlContext  =  yeni  MLContext ( tohum : 1 );

    genel  statik  boşluk  CreateModel ()
    {
        // Veri Yükle
        IDataView  trainingDataView  =  mlContext . Veri . LoadFromTextFile < ModelInput >(
                                        yol : TRAIN_DATA_FILEPATH ,
                                        hasHeader : doğru ,
                                        ayırıcıKarakter : ',' ,
                                        allowQuoting : doğru ,
                                        allowSparse : yanlış );

        // Eğitim hattı oluştur
        IEstimator < ITransformer > trainingPipeline  =  BuildTrainingPipeline ( mlContext );

        // Modelin kalitesini değerlendirin
        Değerlendir ( mlContext , trainingDataView , trainingPipeline );

        // Tren Modeli
        ITransformer  mlModel  =  TrainModel ( mlContext , trainingDataView , trainingPipeline );

        // Modeli kaydet
        SaveModel ( mlContext , mlModel , MODEL_FILEPATH , trainingDataView . Schema );
    }

    genel  statik  IEstimator < ITransformer > BuildTrainingPipeline ( MLContext  mlContext )
    {
        // Ardışık düzen veri dönüşümleriyle veri işlemi yapılandırması
        var  dataProcessPipeline  =  mlContext . Dönüştürür . kategorik . OneHotEncoding ( yeni [] { yeni  InputOutputColumnPair ( " satıcı_kimliği " , " satıcı_kimliği " ), yeni  InputOutputColumnPair ( " ödeme_ türü " , " ödeme_ türü " ) })
                                  . Append ( mlContext . Transforms . Concatenate ( " Özellikler " , yeni [] { " satıcı_kimliği " , " ödeme_türü " , " oran_kodu " , " yolcu_sayısı " , " trip_time_in_secs " , " trip_distance " }));

        // Eğitim algoritmasını ayarla
        var  trainer  =  mlContext . gerileme _ Eğitmenler _ FastTree ( labelColumnName : " fare_amount " , featureColumnName : " Özellikler " );
        var  trainingPipeline  =  dataProcessPipeline . Ekle ( eğitmen );

        dönüş  eğitimiPipeline ;
    }

    genel  statik  ITransformer  TrainModel ( MLContext  mlContext , IDataView  trainingDataView , IEstimator < ITransformer > trainingPipeline )
    {
        konsol . WriteLine ( " =============== Eğitim modeli =============== " );

        ITransformer  modeli  =  trainingPipeline . Sığdır ( trainingDataView );

        konsol . WriteLine ( " =============== Eğitim sürecinin sonu =============== " );
        dönüş  modeli ;
    }

    private  static  void  Evaluate ( MLContext  mlContext , IDataView  trainingDataView , IEstimator < ITransformer > trainingPipeline )
    {
        // Tek veri kümesiyle Çapraz Doğrulama (biri eğitim ve değerlendirme için olmak üzere iki veri kümemiz olmadığı için)
        // modelin doğruluk metriklerini değerlendirmek ve elde etmek için
        konsol . WriteLine ( " =============== Modelin doğruluk ölçümlerini elde etmek için çapraz doğrulama =============== " );
        var  crossValidationResults  =  mlContext . gerileme _ CrossValidate ( trainingDataView , trainingPipeline , numberOfFolds : 5 , labelColumnName : " fare_amount " );
        PrintRegressionFoldsAverageMetrics ( crossValidationResults );
    }
    özel  statik  geçersiz  SaveModel ( MLContext  mlContext , ITransformer  mlModel , string  modelRelativePath , DataViewSchema  modelInputSchema )
    {
        // Eğitilen modeli bir .ZIP dosyasına kaydedin/kaldırın
        konsol . WriteLine ( $" ================ Modeli kaydetme ================ " );
        mlContext . modeli . Kaydet ( mlModel , modelInputSchema , GetAbsolutePath ( modelRelativePath ));
        konsol . WriteLine ( " Model {0} klasörüne kaydedilir " , GetAbsolutePath ( modelRelativePath ));
    }

    genel  statik  dize  GetAbsolutePath ( string  göreceliPath )
    {
        FileInfo  _dataRoot  =  new  FileInfo ( typeof ( Program ). Montaj . Konum );
        string  AssemblyFolderPath  =  _dataRoot . dizin . TamAd ;

        string  fullPath  =  Yol . Birleştir ( assemblyFolderPath , göreceliPath );

         fullPath'i döndür ;
    }

    genel  statik  void  PrintRegressionMetrics ( RegressionMetrics  metrikleri )
    {
        konsol . WriteLine ( $" *************************************************** *** " );
        konsol . WriteLine ( $" * Regresyon modeli için metrikler       " );
        konsol . WriteLine ( $" *------------------------------------------------ --- " );
        konsol . WriteLine ( $" * LossFn: { metrikler . LossFunction : 0. ##} " );
        konsol . WriteLine ( $" * R2 Puanı: { metrikler . RSquared : 0. ##} " );
        konsol . WriteLine ( $" * Mutlak kayıp: { metrikler . MeanAbsoluteError :#.##} " );
        konsol . WriteLine ( $" * Kare kaybı: { metrikler . MeanSquaredError :#.##} " );
        konsol . WriteLine ( $" * RMS kaybı: { metrikler . RootMeanSquaredError :#.##} " );
        konsol . WriteLine ( $" *************************************************** *** " );
    }

    public  static  void  PrintRegressionFoldsAverageMetrics ( IEnumerable < TrainCatalogBase . CrossValidationResult < RegressionMetrics >> crossValidationResults )
    {
        var  L1  =  crossValidationResults . ( r => r . Metrikler . MeanAbsoluteError ); öğesini seçin .  
        var  L2  =  crossValidationResults . ( r => r . Metrikler . MeanSquaredError ); öğesini seçin .  
        var  RMS  =  crossValidationResults . ( r => r . Metrikler . RootMeanSquaredError ) öğesini seçin ;  
        var  kayıpFunction  =  crossValidationResults . ( r => r . Metrics . LossFunction ) öğesini seçin ;  
        var  R2  =  crossValidationResults . ( r => r . Metrikler . RSquared ); öğesini seçin .  

        konsol . WriteLine ( $" *************************************************** ************************************************************ *********** " );
        konsol . WriteLine ( $" * Regresyon modeli için Metrikler       " );
        konsol . WriteLine ( $" *------------------------------------------------ -------------------------------------------------- ------------- " );
        konsol . WriteLine ( $" * Ortalama L1 Kaybı: { L1 . Ortalama (): 0. ###} " );
        konsol . WriteLine ( $" * Ortalama L2 Kaybı: { L2 . Ortalama (): 0. ###}   " );
        konsol . WriteLine ( $" * Ortalama RMS: { RMS . Ortalama (): 0. ###}   " );
        konsol . WriteLine ( $" * Ortalama Kayıp Fonksiyonu: { kayıpFunction . Ortalama (): 0. ###}   " );
        konsol . WriteLine ( $" * Ortalama R-kare: { R2 . Ortalama (): 0. ###}   " );
        konsol . WriteLine ( $" *************************************************** ************************************************************ *********** " );
    }
}
}
