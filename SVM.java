
import java.util.*;

public class SVM {
  double height;
  double weight;
  int label;

  double[][] dataTraining = { { 60, 165, 1 }, { 70, 160, 1 }, { 80, 165, 1 }, { 100, 155, -1 }, { 40, 175, -1 } };
  double[][] datappt = {{1,1,-1},{1,-1,1},{-1,1,1},{-1,-1,-1}};
  double[][] matrixKernel = new double[dataTraining.length][dataTraining.length];
  double[] alphaResult = new double[dataTraining.length];

  SVM(double height, double weight) {
    this.height = height;
    this.weight = weight;
    this.label = 0;
  }

  public void calculateMatrixKernel() {
    // hitung matriks kernel pakai fungsi kernel polinomial d = 2
    for (int i = 0; i < matrixKernel.length; i++) {
      for (int j = 0; j < matrixKernel.length; j++) {
        matrixKernel[i][j] = Math.pow((dataTraining[i][0] * dataTraining[j][0]) + (dataTraining[i][1] * dataTraining[j][1]) + 1, 2);
      }
    }
  }

  public Double getAverage(double [] data) {
    double res = 0;
    for(int i = 0; i< data.length; i++) {
      res += data[i];
    }

    return res/data.length;
  }

  public void sequentialLearning() {
    // init
    double lambda = 1;
    double constanta = 2;
    double gamma = 0;
    double[][] Dij = new double[dataTraining.length][dataTraining.length];
    double[] alpha = new double[dataTraining.length];
    double threshold = 0.00000000001;

    double maxDij = Double.MIN_VALUE;

    double lambdaKuadrat = Math.pow(lambda, 2);

    // cari matriks Dij yang tiap indeks matriks kernel dijumlahkan dengan (constanta + lambda^2)
    // kemudian tiap indeks dikalikan dengan class/label tiap data pada data training
    for (int i = 0; i < matrixKernel.length; i++) {
      for (int j = 0; j < matrixKernel.length; j++) {
        Dij[i][j] += (matrixKernel[i][j] + lambdaKuadrat) * (dataTraining[i][2] * dataTraining[j][2]);
        maxDij = Math.max(Dij[i][j], maxDij);
      }
    }

    // set gamma
    gamma = (0.1d / (50d * maxDij));

    // hitung alpha
    for (int i = 0; i < 2000; i++) {
      double[] Ei = new double[dataTraining.length];
      double[] delta = new double[dataTraining.length];
      double[] newAlpha = new double[dataTraining.length];
      // hitung Ei
      for (int j = 0; j < Dij.length; j++) {
        for (int k = 0; k < alpha.length; k++) {
          Ei[j] += Dij[k][j] * alpha[k];
        }

        delta[j] = Math.min((Math.max(gamma * (1 - Ei[j]), -alpha[j])), constanta - alpha[j]);

        // set nilai baru alfa
        newAlpha[j] = alpha[j] + delta[j];
      }
      System.out.println("Iterasi ke" + i + ": " + Arrays.toString(alpha));
      alpha = newAlpha;
    }
    alphaResult = alpha;
  }

  public void testNewData() {
    double result = 0;
    for (int i = 0; i < dataTraining.length; i++) {
      result += (Math.pow((this.weight * dataTraining[i][0]) + (this.height * dataTraining[i][1]) + 1, 2))
          * (alphaResult[i] * dataTraining[i][2]);
          System.out.println(alphaResult[i]*dataTraining[i][2]);
      System.out.println("the result of iteration " + i + ": " + result);
    }

    this.label = (result >= 0) ? 1 : -1;
  }

  public void predict() {
    System.out.println("========== SVM Classification ===========");

    // cari matriks kernel
    calculateMatrixKernel();

    // cari alfa pakai sequential learning
    sequentialLearning();

    // test data baru
    testNewData();

    System.out.println(this.label);
  }
}