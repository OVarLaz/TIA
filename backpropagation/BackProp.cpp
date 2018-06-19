#include "backprop.h"
#include <time.h>
#include <stdlib.h>


//	inicializa y asigna memoria en un heap
CBackProp::CBackProp(int nl,int *sz,double b,double a):beta(b),alpha(a)
{

	//	conjunto de capas (no) y sus valores
	numl=nl;
	lsize=new int[numl];

	for(int i=0;i<numl;i++){
		lsize[i]=sz[i];
	}

	//	asignar memoria para cada salida por neurona
	out = new double*[numl];

	for( i=0;i<numl;i++){
		out[i]=new double[lsize[i]];
	}

	//	asignar memoria para delta
	delta = new double*[numl];

	for(i=1;i<numl;i++){
		delta[i]=new double[lsize[i]];
	}

	//	asignar memoria para weights (pesos)
	weight = new double**[numl];

	for(i=1;i<numl;i++){
		weight[i]=new double*[lsize[i]];
	}
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			weight[i][j]=new double[lsize[i-1]+1];
		}
	}

	//	asignar memoria para previos weights
	prevDwt = new double**[numl];

	for(i=1;i<numl;i++){
		prevDwt[i]=new double*[lsize[i]];

	}
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			prevDwt[i][j]=new double[lsize[i-1]+1];
		}
	}

	//	preseleccionar y asignar random weights
	srand((unsigned)(time(NULL)));
	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++)
				weight[i][j][k]=(double)(rand())/(RAND_MAX/2) - 1;//32767

	//	inicializar previos weights en 0 para la primera iteracion
	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++)
				prevDwt[i][j][k]=(double)0.0;

}



CBackProp::~CBackProp()
{
	//	liberacion
	for(int i=0;i<numl;i++)
		delete[] out[i];
	delete[] out;

	//	liberacion delta
	for(i=1;i<numl;i++)
		delete[] delta[i];
	delete[] delta;

	//	liberacion weight
	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			delete[] weight[i][j];
	for(i=1;i<numl;i++)
		delete[] weight[i];
	delete[] weight;

	//	liberacion prevDwt
	for(i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			delete[] prevDwt[i][j];
	for(i=1;i<numl;i++)
		delete[] prevDwt[i];
	delete[] prevDwt;

	//	liberacion de informacion de salida
	delete[] lsize;
}

//	funcion sigmoid 
double CBackProp::sigmoid(double in)
{
		return (double)(1/(1+exp(-in)));
}

//	error cuadratico medio
double CBackProp::mse(double *tgt) const
{
	double mse=0;
	for(int i=0;i<lsize[numl-1];i++){
		mse+=(tgt[i]-out[numl-1][i])*(tgt[i]-out[numl-1][i]);
	}
	return mse/2;
}


//	retorna el iesimo valor de la red
double CBackProp::Out(int i) const
{
	return out[numl-1][i];
}

// feed forward para un conjunto de entradas
void CBackProp::ffwd(double *in)
{
	double sum;

	//	asignar contenido para las capas de entrada
	for(int i=0;i<lsize[0];i++)
		out[0][i]=in[i];  // output_from_neuron(i,j) Jth neurona en Ith capa

	//	asignar valor de salida (asinacion) 
	//	para cada neurona usando la funcion sigmoid
	for(i=1;i<numl;i++){				// Para cada capa
		for(int j=0;j<lsize[i];j++){		// Para cada neurona en la capa actual
			sum=0.0;
			for(int k=0;k<lsize[i-1];k++){		// Para la entrada de cada neurona en la capa de procesameinto
				sum+= out[i-1][k]*weight[i][j][k];	// Aplicando los weight para las entradas y agregar para sumar
			}
			sum+=weight[i][j][lsize[i-1]];		// Aplicando bias
			out[i][j]=sigmoid(sum);				// Aplicando funcion sigmoid 
		}
	}
}


//	retropropagar errores a las salidas
//	hasta la primera capa oculta
void CBackProp::bpgt(double *in,double *tgt)
{
	double sum;

	//	actualizar los valores de salida para cada neurona
	ffwd(in);

	//	encontrar delta para la capa de salida
	for(int i=0;i<lsize[numl-1];i++){
		delta[numl-1][i]=out[numl-1][i]*
		(1-out[numl-1][i])*(tgt[i]-out[numl-1][i]);
	}

	//	encontrar delta para las capas escondidas	
	for(i=numl-2;i>0;i--){
		for(int j=0;j<lsize[i];j++){
			sum=0.0;
			for(int k=0;k<lsize[i+1];k++){
				sum+=delta[i+1][k]*weight[i+1][k][j];
			}
			delta[i][j]=out[i][j]*(1-out[i][j])*sum;
		}
	}

	//	aplicando momentum ( nada si alpha=0 )
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				weight[i][j][k]+=alpha*prevDwt[i][j][k];
			}
			weight[i][j][lsize[i-1]]+=alpha*prevDwt[i][j][lsize[i-1]];
		}
	}

	//	ajustando weights usando gradiente descendiente
	for(i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				prevDwt[i][j][k]=beta*delta[i][j]*out[i-1][k];
				weight[i][j][k]+=prevDwt[i][j][k];
			}
			prevDwt[i][j][lsize[i-1]]=beta*delta[i][j];
			weight[i][j][lsize[i-1]]+=prevDwt[i][j][lsize[i-1]];
		}
	}
}

