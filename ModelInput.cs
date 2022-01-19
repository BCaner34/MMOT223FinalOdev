//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using Microsoft.ML.Data;

namespace PremLig.PredictionML.Model.DataModels
{
    public class ModelInput
{

    [ColumnName(@"HomeTeam"), LoadColumn(1)]
    public string HomeTeam { get; set; }

    [ColumnName(@"AwayTeam"), LoadColumn(2)]
    public string AwayTeam { get; set; }

    [ColumnName(@"FTHomeGol"), LoadColumn(3)]
    public float FTHomeGol { get; set; }

    [ColumnName(@"FTAwayGol"), LoadColumn(4)]
    public float FTAwayGol { get; set; }

    [ColumnName(@"HomeShot"), LoadColumn(5)]
    public float HomeShot { get; set; }

    [ColumnName(@"AwayShot"), LoadColumn(6)]
    public float AwayShot { get; set; }

    [ColumnName(@"HomeCorner"), LoadColumn(7)]
    public float HomeCorner { get; set; }

    [ColumnName(@"AwayCorner"), LoadColumn(8)]
    public float AwayCorner { get; set; }
    }
}
