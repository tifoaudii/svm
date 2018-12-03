import java.util.*;

public class main {
  public static void main(String [] args) {
    Scanner in = new Scanner(System.in);
    System.out.print("Masukkan tinggi \t: ");
    double height = in.nextDouble();
    System.out.print("Masukkan berat \t\t: ");
    double weight = in.nextDouble();

    //init object svm
    SVM svm = new SVM(height,weight);
    svm.predict();
    in.close();
  }
}