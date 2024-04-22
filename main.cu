#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cblas.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <bits/stdc++.h>

// Define the payoff matrix (assuming Hawk-Dove game)
#define FULL_PAYOFF 20
#define HAWK_VS_DOVE_PAYOFF 15
#define DOVE_VS_HAWK_PAYOFF 5
#define DOVE_VS_DOVE_PAYOFF 10

__global__ void hawkDoveKernel(int *strategies, int *score, int numBirds)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < numBirds)
    {
        int my_strategy = strategies[idx];
        int opponent_strategy = strategies[(idx + 1) % numBirds]; // Simple opponent selection

        // Determine score based on strategies
        if (my_strategy == 1 && opponent_strategy == 1)
        { // Both players choose Hawk
          // The cost is paid prior to calling this method.
        }
        else if (my_strategy == 1 && opponent_strategy == 0)
        { // Hawk vs Dove
            score[idx] = HAWK_VS_DOVE_PAYOFF;
        }
        else if (my_strategy == 0 && opponent_strategy == 1)
        { // Dove vs Hawk
            score[idx] = DOVE_VS_HAWK_PAYOFF;
        }
        else if (my_strategy == 0 && opponent_strategy == 0)
        { // Both players choose Dove
            score[idx] = DOVE_VS_DOVE_PAYOFF;
        }
        else
        {
            score[idx] = FULL_PAYOFF;
        }
    }
}

enum struct Strategy : uint16_t
{
    Share = 0,
    Steal,
    Bush
};

enum struct LifeCycle : uint16_t
{
    Live = 0,
    Reproduce,
    Die
};

class Creature
{
public:
    Strategy strategy;
    int score = 0;
    LifeCycle lifecycle = LifeCycle::Live;
};

class Bird : Creature
{
public:
    Bird(Strategy input)
    {
        strategy = input;
        score = 10;
    }
    Strategy getStrategy()
    {
        return strategy;
    }
    void setStrategy(const Strategy &input)
    {
        strategy = input;
    }
    int getScore()
    {
        return score;
    }
    void setScore(const int &input)
    {
        score = input;
    }
    LifeCycle getLifeCycle()
    {
        return lifecycle;
    }
    void setLifeCycle(const LifeCycle &input)
    {
        lifecycle = input;
    }
};

int strategyToInt(Strategy input)
{
    if (input == Strategy::Steal)
    {
        return 0;
    }
    else if (input == Strategy::Share)
    {
        return 1;
    }
    else
    {
        return 2;
    }
}

void whatAmI(Bird &testBird)
{
    if (testBird.getStrategy() == Strategy::Steal)
    {
        std::cout << "I am a Hawk." << std::endl;
    }
    else if (testBird.getStrategy() == Strategy::Share)
    {
        std::cout << "I am a Dove." << std::endl;
    }
    else
    {
        std::cout << "I am Unknown." << std::endl;
    }
}

void whatIsMyLifeCycle(Bird &testBird)
{
    if (testBird.getLifeCycle() == LifeCycle::Live)
    {
        std::cout << "I will Live." << std::endl;
    }
    else if (testBird.getLifeCycle() == LifeCycle::Die)
    {
        std::cout << "I will Die." << std::endl;
    }
    else if (testBird.getLifeCycle() == LifeCycle::Reproduce)
    {
        std::cout << "I will Reproduce." << std::endl;
    }
}

void countBirds(std::vector<Bird> &birdArray, int arraySize, int &hawks,
                int &doves)
{
    doves = 0;
    hawks = 0;

    for (int i = 0; i < arraySize; i++)
    {
        if (birdArray[i].getStrategy() == Strategy::Share)
            doves++;
        else
            hawks++;
    }
}

void contest(Bird &first, Bird &second, int v, int c)
{
    // Test logs left in.
    if (first.getStrategy() == Strategy::Share && second.getStrategy() == Strategy::Share)
    {
        // std::cout << "Both are Doves, share resources \n";
        first.setScore(first.getScore() + (v / 2));
        second.setScore(second.getScore() + (v / 2));
    }
    else if (first.getStrategy() == Strategy::Steal && second.getStrategy() == Strategy::Steal)
    {
        // Cost is paid in the day cycle
        // If there's an update to move the cost into this method, this'll be used.
        // {
        //     if (rand() % 2 == 1)
        //     {
        //         // std::cout << "Both are Hawks, but first won \n";
        //         first.setScore(first.getScore() + v);
        //         second.setScore(second.getScore() - c);
        //     }
        //     else
        //     {
        //         // std::cout << "Both are Hawks, but 2nd won \n";
        //         first.setScore(first.getScore() - c);
        //         second.setScore(second.getScore() + v);
        //     }
        // }
    }
    else if (first.getStrategy() == Strategy::Share && second.getStrategy() == Strategy::Steal)
    {
        // std::cout << "Dove Meets Hawk \n";
        first.setScore(first.getScore() + (v * .25));
        second.setScore(second.getScore() + (v * .75));
    }
    else if (first.getStrategy() == Strategy::Steal && second.getStrategy() == Strategy::Share)
    {
        // std::cout << "Hawk Meets Dove \n";
        first.setScore(first.getScore() + (v * .75));
        second.setScore(second.getScore() + (v * .25));
    }
    else
    {
        // std::cout << "There's a bush \n";
        first.setScore(first.getScore() + v);
        second.setScore(second.getScore() + v);
    }
}

void vectorAddMultiples(std::vector<Bird> &array, int size, Strategy toAdd)
{
    for (int i = 0; i < size; i++)
    {
        Bird newBird(toAdd);
        array.push_back(newBird);
    }
}

void runTestGpuKernel(std::vector<Bird> &birdVector, int bushes)
{
    // The cost of life
    for (unsigned i = 0; i < birdVector.size(); ++i)
        birdVector[i].setScore(birdVector[i].getScore() - 10);

    int diff = (bushes * 2) - birdVector.size();
    if (diff > 0)
        vectorAddMultiples(birdVector, diff, Strategy::Bush);

    // Shuffle to randomize the contests
    std::shuffle(std::begin(birdVector), std::end(birdVector), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

    // Initialize game parameters
    int numOfBirds = birdVector.size(); // Number of birds
    int strategies[numOfBirds];         // Array to hold strategies (0 for Dove, 1 for Hawk, 2 for Bush)
    int payoffs[numOfBirds];            // Array to hold payoffs

    // Initialize strategies randomly (for demonstration)
    for (int i = 0; i < numOfBirds; ++i)
    {
        strategies[i] = strategyToInt(birdVector[i].getStrategy());
    }

    // Allocate device memory
    int *d_strategies;
    int *d_payoffs;
    checkCudaErrors(cudaMalloc(&d_strategies, numOfBirds * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_payoffs, numOfBirds * sizeof(int)));

    // Copy strategies to device
    checkCudaErrors(cudaMemcpy(d_strategies, strategies, numOfBirds * sizeof(int), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (numOfBirds + blockSize - 1) / blockSize;

    double cumulativeTime = 0.0;

    {
        cudaEvent_t start_, stop_;
        checkCudaErrors(cudaEventCreate(&start_));
        checkCudaErrors(cudaEventCreate(&stop_));
        checkCudaErrors(cudaEventRecord(start_));

        // Launch kernel
        hawkDoveKernel<<<numBlocks, blockSize>>>(d_strategies, d_payoffs, numOfBirds);

        checkCudaErrors(cudaEventRecord(stop_));
        checkCudaErrors(cudaEventSynchronize(stop_));

        // Copy payoffs from device
        checkCudaErrors(cudaMemcpy(payoffs, d_payoffs, numOfBirds * sizeof(int), cudaMemcpyDeviceToHost));

        float milliseconds = 0.0f;

        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start_, stop_));
        cumulativeTime += milliseconds;
        checkCudaErrors(cudaEventDestroy(start_));
        checkCudaErrors(cudaEventDestroy(stop_));
    }

    // Display payoffs (for demonstration)
    // printf("Payoffs:\n");
    // for (int i = 0; i < numOfBirds; ++i)
    // {
    //     printf("Player %d: %d\n", i, payoffs[i]);
    // }

    for (int i = 0; i < numOfBirds; ++i)
    {
        birdVector[i].setScore(payoffs[i]);
    }

    for (int i = birdVector.size() - 1; i >= 0; i--)
    {
        if (birdVector[i].getStrategy() == Strategy::Bush)
        {
            birdVector.erase(birdVector.begin() + i);
        }
    }

    // Free device memory
    checkCudaErrors(cudaFree(d_strategies));
    checkCudaErrors(cudaFree(d_payoffs));
}

void DayCycle(std::vector<Bird> &birdVector, int bushes)
{
    // The cost of life
    for (unsigned i = 0; i < birdVector.size(); ++i)
        birdVector[i].setScore(birdVector[i].getScore() - 10);

    int diff = (bushes * 2) - birdVector.size();
    if (diff > 0)
        vectorAddMultiples(birdVector, diff, Strategy::Bush);

    // Shuffle to randomize the contests
    std::shuffle(std::begin(birdVector), std::end(birdVector), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

    for (int j = 0; j < (bushes * 2); j += 2)
    {
        contest(birdVector[j], birdVector[j + 1], 20, 0);
    }

    for (int i = birdVector.size() - 1; i >= 0; i--)
    {
        if (birdVector[i].getStrategy() == Strategy::Bush)
        {
            birdVector.erase(birdVector.begin() + i);
        }
    }
}

void NightCycle(std::vector<Bird> &array)
{
    int babyDoves = 0;
    int babyHawks = 0;
    int testValue = 0;

    for (int i = 0; i < array.size(); i++)
    {
        array[i].setLifeCycle(LifeCycle::Live);

        int chance = rand() % 10 + 1;

        testValue = array[i].getScore();
        if (testValue < 10)
        {
            if (testValue < chance)
                array[i].setLifeCycle(LifeCycle::Die);
        }
        else if (testValue >= 20)
        {
            array[i].setScore(testValue - 10);
            if (array[i].getStrategy() == Strategy::Steal)
            {
                babyHawks++;
            }
            else
            {
                babyDoves++;
            }
        }
        else
        {
            if (testValue > (chance + 10))
            {
                array[i].setScore(testValue - 10);
                if (array[i].getStrategy() == Strategy::Steal)
                {
                    babyHawks++;
                }
                else
                {
                    babyDoves++;
                }
            }
        }
    }

    // Test log
    // std::cout << " Adding babies, " << babyDoves << " doves and " << babyHawks << " hawks \n";
    vectorAddMultiples(array, babyDoves, Strategy::Share);
    vectorAddMultiples(array, babyHawks, Strategy::Steal);

    for (int i = array.size() - 1; i >= 0; i--)
    {
        if (array[i].getLifeCycle() == LifeCycle::Die)
        {
            array.erase(array.begin() + i);
        }
    }
}

int runCpu()
{
    std::vector<Bird> arr;
    int bushes = 60;
    int hawks = 10;
    int doves = 10;
    int generations = 100;

    vectorAddMultiples(arr, doves, Strategy::Share);
    vectorAddMultiples(arr, hawks, Strategy::Steal);

    int hawksval = 0;
    int dovesval = 0;

    std::ofstream myfile;
    myfile.open("cpuoutput.csv");
    myfile << "dove,hawk\n";

    auto start = std::chrono::high_resolution_clock::now();

    // Loop function
    for (int i = 1; i <= generations; i++)
    {
        // Count birds and output birds and generation
        countBirds(arr, arr.size(), hawksval, dovesval);
        std::cout << "Generation : " << i << ". # of bushes : " << bushes << ". # of Hawks : " << hawksval << " # of Doves : " << dovesval << std::endl;

        myfile << dovesval << "," << hawksval << "\n";

        // Allow life to happen
        DayCycle(arr, bushes);
        NightCycle(arr);
    }

    auto stop = std::chrono::high_resolution_clock::now();

    myfile.close();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Average Time taken by function: "
              << duration.count() / generations << " microseconds" << std::endl;
    return 0;
}

void printTestInfo(std::vector<Bird> &arr)
{
    std::cout << "Number of Birds " << arr.size() << std::endl;

    std::cout << "Birds: ";

    for (unsigned i = 0; i < arr.size(); ++i)
        std::cout << ' ' << arr[i].getScore();
    std::cout << '\n';
}

void runGPU()
{
    std::vector<Bird> arr;
    int bushes = 60;
    int hawks = 10;
    int doves = 10;
    int generations = 100;

    vectorAddMultiples(arr, doves, Strategy::Share);
    vectorAddMultiples(arr, hawks, Strategy::Steal);

    int hawksval = 0;
    int dovesval = 0;

    std::ofstream myfile;
    myfile.open("gpuoutput.csv");
    myfile << "dove,hawk\n";

    auto start = std::chrono::high_resolution_clock::now();

    // Loop function
    for (int i = 1; i <= generations; i++)
    {
        // Count birds and output birds and generation
        countBirds(arr, arr.size(), hawksval, dovesval);
        std::cout << "Generation : " << i << ". # of bushes : " << bushes << ". # of Hawks : " << hawksval << " # of Doves : " << dovesval << std::endl;

        myfile << dovesval << "," << hawksval << "\n";

        // Allow life to happen
        runTestGpuKernel(arr, bushes);
        NightCycle(arr);
    }

    auto stop = std::chrono::high_resolution_clock::now();

    myfile.close();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Average Time taken by function: "
              << duration.count() / generations << " microseconds" << std::endl;
}

int main()
{
    std::cout << "Runing CPU" << std::endl;

    runCpu();

    std::cout << "\n Runing GPU" << std::endl;
    runGPU();

    return 0;
}
