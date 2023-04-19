template <int INPUTS, int OUTPUTS>
struct trainingdata
{
    struct sample
    {
        float inputs[INPUTS]{};
        float outputs[OUTPUTS]{};
    };
};

template <int LAYERS>
class B
{
public:
    template <int INPUTS, int OUTPUTS>
    float pulse(typename trainingdata<INPUTS, OUTPUTS>::sample& sample)
    {
        return sample.inputs[INPUTS - 1] + sample.outputs[OUTPUTS - 1];
    }
};