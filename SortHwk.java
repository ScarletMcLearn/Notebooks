import java.util.Arrays;

public class SortHwk
{
    public static int[][] sort(int mat[][]){
        /** 
        /* This function sorts the elements of an array. 
        /* After sorting, the elements will be arranged in descending order. 
        /* The input array is stored at mat[1][:]. Its attached label is stored at mat[0][:].
        /*
        /* Return:
        /*     the sorted array and its corresponding label
        /* Locally, the sorted array is stored at newmat[1][:]. Its attached label is stored at newmat[0][:].
        */
        int i,j,tmp,tmpLabel,index=1;
        int newmat[][]=new int[mat.length][mat[0].length];
 
        for(i=0;i<mat.length;i++){                    //copy the input matrix
            for(j=0;j<mat[0].length;j++){
            newmat[i][j]=mat[i][j];
            }
        }
        for(i=0;i<newmat[0].length;i++){
            tmp=newmat[1][i];
            tmpLabel=newmat[0][i];
            for(j=i+1;j<newmat[0].length;j++){
                if(newmat[1][j]>tmp){                 //find the biggest number
                    tmp=newmat[1][j];
                    tmpLabel=newmat[0][j];
                    index=j;
                    newmat[1][index]=newmat[1][i];    //swap
                    newmat[1][i]=tmp;
                    newmat[0][index]=newmat[0][i];    //swap the Label of arr, too
                    newmat[0][i]=tmpLabel;
                }
            }
        }
        return newmat;
    }
    
    public static int[] RanArrayGen(int randarr[], int MaxNum){
        /** 
        /* This function creats an array which contains uniformly distributed random numbers(integer). 
        /* The integers are within the range [1,MaxNum]. 
        /*
        /* Return:
        /*     the array of random numbers(integer)
        */
        int i;        
        for(i=0;i<randarr.length;i++){                //create an array which contains random numbers ranging from 1 to 42
            randarr[i]=(int)(Math.random()*MaxNum+1);
        }
        return randarr;
    }

    public static int[][] RanArraystats(int randarr[],int MaxNum){
        /** 
        /* This function will see how the random numers in "randarr" are distributed.
        /* 
        /*
        /* Return:
        /*     Matrix (size: MaxNum X MaxNum)
        */
        int i,j;
        int mat[][]=new int[MaxNum][MaxNum];
        for(i=0;i<MaxNum;i++){                  //see how the numbers in the array "randarr" are distributed.
            mat[0][i]=i+1;
            mat[1][i]=0;
            for(j=0;j<randarr.length;j++){
                if(randarr[j]==i+1){mat[1][i]++;}
            }
        }
        //System.out.println(Arrays.toString(mat[0]));
        //System.out.println(Arrays.toString(mat[1]));
        return mat; 
    }

    public static void PrintResults(int mat[][], int newmat[][]){
        /** 
        /* This function prints the final result.
        /* 
        /* Return:
        /*     Nothing
        */
        int i,j;

        System.out.printf("original|    data | sorted|   data \n");    // output results
        System.out.printf("---------------------------------- \n");    // output results
        for(i=0;i<mat[0].length;i++){
            System.out.printf("%7d | %7d | %5d | %6d \n",mat[0][i],mat[1][i],newmat[0][i],newmat[1][i]);
        }
        
    }
    
    public static void main(String CmdLineArgs[]){
        int i,j;
        int randarr[]=new int[100000];
        int mat[][]=new int[2][42];
        int newmat[][]=new int[2][42];
        
        RanArrayGen(randarr,mat[0].length);           // create an array that stores random numbers
        mat=RanArraystats(randarr,mat[0].length);     // learn how many times that a label has shown in the created array
        newmat=sort(mat);                             // sort the (label, number of times) matrix
        PrintResults(mat,newmat);                     // output the final result
    }
}