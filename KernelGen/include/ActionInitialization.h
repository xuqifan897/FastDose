#ifndef __ACTIONINITIALIZATION_H__
#define __ACTIONINITIALIZATION_H__

#include "G4VUserActionInitialization.hh"

namespace fastdose {
    class ActionInitialization : public G4VUserActionInitialization {
    public:
        ActionInitialization() = default;
        ~ActionInitialization() override = default;

        void BuildForMaster() const override;
        void Build() const override;
    };
}

#endif