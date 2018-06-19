
#ifndef backprop_h
#define backprop_h

#include<assert.h>
#include<iostream.h>
#include<stdio.h>
#include<math.h>


class CBackProp{

//	salida para cada neurona
	double **out;

//	valor de error delta para cada neurona
	double **delta;

//	vector de weights para cada neurona
	double ***weight;

//	no hay capas en la red
//	incluyendo capa de entrada
	int numl;

//	vector de elementos de numeros por tamanio
//	para cada capa
	int *lsize;

//	aprendizaje
	double beta;

//	parametro momentum 
	double alpha;

//	almacenamiento para modo weight-change 
//	para parametro previo
	double ***prevDwt;

//	funcion de acumulamiento
	double sigmoid(double in);

public:

	~CBackProp();

//	inicializacion y almacenamiento de memoria
	CBackProp(int nl,int *sz,double b,double a);

//	retropropagacion del error para un conjunto de entradas
	void bpgt(double *in,double *tgt);

//	activacion feed forwards para un conjunto de entradas
	void ffwd(double *in);

//	retorna el error medio cuadratico de la rd
	double mse(double *tgt) const;	
	
//	retorna la iesima salida de la red
	double Out(int i) const;
};

#endif