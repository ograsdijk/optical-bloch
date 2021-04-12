module TlFOpticalBloch

import Waveforms
include("P2F1.jl")
include("P2F1_J12.jl")

function square_wave(t::Float64, frequency::Float64, phase::Float64)
    0.5.*(1 .+ Waveforms.squarewave(2*pi.*frequency.*t .+ phase))
end

end # module
