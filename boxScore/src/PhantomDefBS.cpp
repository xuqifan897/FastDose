#include <vector>
#include <string>

#include "globals.hh"
#include "G4SystemOfUnits.hh"

#include "PhantomDefBS.h"
#include "argparseBS.h"

bs::GeomDef* bs::GD = nullptr;

bs::GeomDef::GeomDef()
{
    this->layers.push_back(std::make_tuple("adipose", 0.8*cm));
    this->layers.push_back(std::make_tuple("muscle", 0.8*cm));
    this->layers.push_back(std::make_tuple("bone", 0.8*cm));
    this->layers.push_back(std::make_tuple("muscle", 0.8*cm));
    this->layers.push_back(std::make_tuple("lung", 4.8*cm));
    this->layers.push_back(std::make_tuple("muscle", 0.8*cm));
    this->layers.push_back(std::make_tuple("bone", 0.8*cm));
    this->layers.push_back(std::make_tuple("adipose", 0.8*cm));
    this->layers.push_back(std::make_tuple("bone", 0.8*cm));
    this->layers.push_back(std::make_tuple("muscle", 0.8*cm));
    this->layers.push_back(std::make_tuple("adipose", 0.8*cm));
}

void bs::GeomDef::display()
{
    // display spectrum information first
    float EFluenceTotal = 0.;
    for (auto& it : Spec)
        EFluenceTotal += std::get<1>(it);
    std::cout << "The sum of total energy: " << std::setw(4) << 
        std::setfill('0') << EFluenceTotal << std::endl << std::setfill(' ');
    std::vector<float> photonCounts(Spec.size());

    float fluenceTotal = 0.f;
    for (int i=0; i<Spec.size(); i++) {
        fluenceTotal += std::get<1>(Spec[i]);
    }
    for (int i=0; i<Spec.size(); i++) {
        std::get<1>(Spec[i]) /= fluenceTotal;
    }
    int nParticles = (*bs::vm)["nParticles"].as<int>();
    int prev = 0;
    for (int i=0; i<Spec.size(); i++) {
        std::get<2>(Spec[i]) = int(std::round(std::get<1>(Spec[i]) * nParticles));
        std::get<3>(Spec[i]) = std::get<2>(Spec[i]) + prev;
        prev += std::get<2>(Spec[i]);
    }

    // display the spectrum information
    std::cout << std::setw(20) << std::left << "Energy (MeV)" << std::setw(20) << 
        std::left << "Fluence" << std::setw(20) << std::left << "Number Fluence" << 
        std::setw(20) << "Cummu Number Fluence" << std::endl;
    for (int i=0; i<Spec.size(); i++)
        std::cout << std::setw(20) << std::setprecision(2) << std::get<0>(Spec[i]) << 
            std::setw(20) << std::setprecision(4) << std::get<1>(Spec[i]) <<
            std::setw(20) << std::get<2>(Spec[i]) << std::setw(20) << std::get<3>(Spec[i]) << std::endl;
    std::cout << std::endl;

    // then display phantom information
    for (int i=0; i<this->layers.size(); i++)
    {
        std::string& mat = std::get<0>(this->layers[i]);
        float thickness = std::get<1>(this->layers[i]);
        std::cout << "material:" << std::setw(15) << std::right << mat << ", ";
        std::cout << "thickness:" << std::setw(14) << std::right << thickness/cm << "cm" << std::endl;
    }
}

std::vector<std::tuple<float, float, int, int>> bs::Spec{
    {0.20, 0.001, 0, 0},
    {0.30, 0.010, 0, 0},
    {0.40, 0.020, 0, 0},
    {0.50, 0.030, 0, 0},
    {0.60, 0.068, 0, 0},
    {0.80, 0.090, 0, 0},
    {1.00, 0.101, 0, 0},
    {1.25, 0.100, 0, 0},
    {1.50, 0.131, 0, 0},
    {2.00, 0.188, 0, 0},
    {3.00, 0.140, 0, 0},
    {4.00, 0.090, 0, 0},
    {5.00, 0.030, 0, 0},
    {6.00, 0.005, 0, 0}};