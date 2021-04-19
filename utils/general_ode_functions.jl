using Waveforms
function sine_wave(t, frequency, phase)
    0.5.*(1+sin(2*pi.*frequency.*t .+ phase))
end

function gaussian_2d(x::Float64, y::Float64, a::Float64, μx::Float64, μy::Float64, σx::Float64, σy::Float64)::Float64
    a.*exp(.- ((x.-μx).^2 ./ (2 .* σx.*σx) + (y.-μy).^2 ./ (2 .* σy.*σy)))
end

function square_wave(t::Float64, frequency::Float64, phase::Float64)
    0.5.*(1 .+ Waveforms.squarewave(2*pi.*frequency.*t .+ phase))
end