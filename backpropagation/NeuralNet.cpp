
#include "BackProp.h"

#include "backprop.h"

int main(int argc, char* argv[])
{


	// XOR traing data
	double data[][4]={
				0,	0,	0,	0,
				0,	0,	1,	1,
				0,	1,	0,	1,
				0,	1,	1,	0,
				1,	0,	0,	1,
				1,	0,	1,	0,
				1,	1,	0,	0,
				1,	1,	1,	1 };

	// test data	
	double testData[][3]={
								0,      0,      0,
                                0,      0,      1,
                                0,      1,      0,
                                0,      1,      1,
                                1,      0,      0,
                                1,      0,      1,
                                1,      1,      0,
                                1,      1,      1};

	
	// red con 4 capas 3,3,3, y 1 neurona respectivamente,
	//la primera capa es la capa de entrada (tomando valores)
	// y tiene que ser el mismo tamanio como el numero de parametros de entrada
	int numLayers = 4, lSz[4] = {3,3,2,1};

	
	// Aprendizaje - beta
	// momentum - alpha
	// Threshhold - thresh 
	double beta = 0.3, alpha = 0.1, Thresh =  0.00001;

	
	// maximo numero de iteraciones durante entrenamiento
	long num_iter = 2000000;

	
	// Creado red
	CBackProp *bp = new CBackProp(numLayers, lSz, beta, alpha);
	
	cout<< endl <<  "Entrenando red...." << endl;	
	for (long i=0; i<num_iter ; i++)
	{
		
		bp->bpgt(data[i%8], &data[i%8][3]);

		if( bp->mse(&data[i%8][3]) < Thresh) {
			cout << endl << "Red entrenada. Valor limite archivado en " << i << " iteraciones." << endl;
			cout << "MSE:  " << bp->mse(&data[i%8][3]) 
				 <<  endl <<  endl;
			break;
		}
		if ( i%(num_iter/10) == 0 )
			cout<<  endl <<  "MSE:  " << bp->mse(&data[i%8][3]) 
				<< "... Entrenando..." << endl;

	}
	
	if ( i == num_iter )
		cout << endl << i << " iteraciones completadas..." 
		<< "MSE: " << bp->mse(&data[(i-1)%8][3]) << endl;  	

	cout<< "Usando red de entrenamiento para hacer predicciones en test data...." << endl << endl;	
	for ( i = 0 ; i < 8 ; i++ )
	{
		bp->ffwd(testData[i]);
		cout << testData[i][0]<< "  " << testData[i][1]<< "  "  << testData[i][2]<< "  " << bp->Out(0) << endl;
	}

	return 0;
}



