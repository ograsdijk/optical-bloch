function Lindblad_rhs!(du, ρ, p, t)
	@inbounds begin
		Ω1 = p[1]
		νp1 = p[2]
		Δ = p[3]
		Ω1ᶜ = conj(Ω1)
		Px1 = sine_wave(t, νp1, 4.71238898038469)
		Pz1 = sine_wave(t, νp1, 1.5707963267948966)
		norm1 = sqrt(Px1^2+Pz1^2)
		Px1 /= norm1
		Pz1 /= norm1
		du[1,2] = -1.0*1im*(-0.353552889862801*Ω1ᶜ*ρ[1,13]*Px1 - 139741.094436646*ρ[1,2])
		du[1,3] = -1.0*1im*(0.5*Ω1ᶜ*ρ[1,13]*Pz1 - 139730.395614624*ρ[1,3])
		du[1,4] = -1.0*1im*(0.353553895977235*Ω1ᶜ*ρ[1,13]*Px1 - 139719.693466187*ρ[1,4])
		du[1,5] = -1.0*1im*(-0.205619589808608*Ω1ᶜ*ρ[1,13]*Px1 - 1245271.96733093*ρ[1,5])
		du[1,6] = -1.0*1im*(0.29078878665932*Ω1ᶜ*ρ[1,13]*Pz1 - 1245272.14602661*ρ[1,6])
		du[1,7] = -1.0*1im*(0.205617856697416*Ω1ᶜ*ρ[1,13]*Px1 - 1245272.32510376*ρ[1,7])
		du[1,8] = 1336642.61555481*1im*ρ[1,8]
		du[1,9] = 1336632.31471252*1im*ρ[1,9]
		du[1,10] = 1336622.01324463*1im*ρ[1,10]
		du[1,11] = 1336611.71130371*1im*ρ[1,11]
		du[1,12] = 1336601.40881348*1im*ρ[1,12]
		du[1,13] = -5026548.21497941*ρ[1,13] - 1.0*1im*(-Δ*ρ[1,13] - 0.353552889862801*Ω1*ρ[1,2]*Px1 + 0.5*Ω1*ρ[1,3]*Pz1 + 0.353553895977235*Ω1*ρ[1,4]*Px1 - 0.205619589808608*Ω1*ρ[1,5]*Px1 + 0.29078878665932*Ω1*ρ[1,6]*Pz1 + 0.205617856697416*Ω1*ρ[1,7]*Px1 - 139730.395614624*ρ[1,13])
		du[2,1] = -1.0*1im*(0.353552889862801*Ω1*ρ[13,1]*Px1 + 139741.094436646*ρ[2,1])
		du[2,2] = 2504066.14187607*ρ[13,13] - 1.0*1im*(0.353552889862801*Ω1*ρ[13,2]*Px1 - 0.353552889862801*Ω1ᶜ*ρ[2,13]*Px1)
		du[2,3] = -1.0*1im*(0.353552889862801*Ω1*ρ[13,3]*Px1 + 0.5*Ω1ᶜ*ρ[2,13]*Pz1 + 10.6988220214844*ρ[2,3])
		du[2,4] = -1.0*1im*(0.353552889862801*Ω1*ρ[13,4]*Px1 + 0.353553895977235*Ω1ᶜ*ρ[2,13]*Px1 + 21.4009704589844*ρ[2,4])
		du[2,5] = -1.0*1im*(0.353552889862801*Ω1*ρ[13,5]*Px1 - 0.205619589808608*Ω1ᶜ*ρ[2,13]*Px1 - 1105530.87289429*ρ[2,5])
		du[2,6] = -1.0*1im*(0.353552889862801*Ω1*ρ[13,6]*Px1 + 0.29078878665932*Ω1ᶜ*ρ[2,13]*Pz1 - 1105531.05158997*ρ[2,6])
		du[2,7] = -1.0*1im*(0.353552889862801*Ω1*ρ[13,7]*Px1 + 0.205617856697416*Ω1ᶜ*ρ[2,13]*Px1 - 1105531.23066711*ρ[2,7])
		du[2,8] = -1.0*1im*(0.353552889862801*Ω1*ρ[13,8]*Px1 - 1196901.52111816*ρ[2,8])
		du[2,9] = -1.0*1im*(0.353552889862801*Ω1*ρ[13,9]*Px1 - 1196891.22027588*ρ[2,9])
		du[2,10] = -1.0*1im*(0.353552889862801*Ω1*ρ[13,10]*Px1 - 1196880.91880798*ρ[2,10])
		du[2,11] = -1.0*1im*(0.353552889862801*Ω1*ρ[13,11]*Px1 - 1196870.61686707*ρ[2,11])
		du[2,12] = -1.0*1im*(0.353552889862801*Ω1*ρ[13,12]*Px1 - 1196860.31437683*ρ[2,12])
		du[2,13] = -5026548.21497941*ρ[2,13] - 1.0*1im*(-Δ*ρ[2,13] - 0.353552889862801*Ω1*ρ[2,2]*Px1 + 0.5*Ω1*ρ[2,3]*Pz1 + 0.353553895977235*Ω1*ρ[2,4]*Px1 - 0.205619589808608*Ω1*ρ[2,5]*Px1 + 0.29078878665932*Ω1*ρ[2,6]*Pz1 + 0.205617856697416*Ω1*ρ[2,7]*Px1 + 0.353552889862801*Ω1*ρ[13,13]*Px1 + 10.6988220214844*ρ[2,13])
		du[3,1] = -1.0*1im*(-0.5*Ω1*ρ[13,1]*Pz1 + 139730.395614624*ρ[3,1])
		du[3,2] = -1.0*1im*(-0.5*Ω1*ρ[13,2]*Pz1 - 0.353552889862801*Ω1ᶜ*ρ[3,13]*Px1 - 10.6988220214844*ρ[3,2])
		du[3,3] = 2504073.23484697*ρ[13,13] - 1.0*1im*(-0.5*Ω1*ρ[13,3]*Pz1 + 0.5*Ω1ᶜ*ρ[3,13]*Pz1)
		du[3,4] = -1.0*1im*(-0.5*Ω1*ρ[13,4]*Pz1 + 0.353553895977235*Ω1ᶜ*ρ[3,13]*Px1 + 10.7021484375*ρ[3,4])
		du[3,5] = -1.0*1im*(-0.5*Ω1*ρ[13,5]*Pz1 - 0.205619589808608*Ω1ᶜ*ρ[3,13]*Px1 - 1105541.57171631*ρ[3,5])
		du[3,6] = -1.0*1im*(-0.5*Ω1*ρ[13,6]*Pz1 + 0.29078878665932*Ω1ᶜ*ρ[3,13]*Pz1 - 1105541.75041199*ρ[3,6])
		du[3,7] = -1.0*1im*(-0.5*Ω1*ρ[13,7]*Pz1 + 0.205617856697416*Ω1ᶜ*ρ[3,13]*Px1 - 1105541.92948914*ρ[3,7])
		du[3,8] = -1.0*1im*(-0.5*Ω1*ρ[13,8]*Pz1 - 1196912.21994019*ρ[3,8])
		du[3,9] = -1.0*1im*(-0.5*Ω1*ρ[13,9]*Pz1 - 1196901.9190979*ρ[3,9])
		du[3,10] = -1.0*1im*(-0.5*Ω1*ρ[13,10]*Pz1 - 1196891.61763*ρ[3,10])
		du[3,11] = -1.0*1im*(-0.5*Ω1*ρ[13,11]*Pz1 - 1196881.31568909*ρ[3,11])
		du[3,12] = -1.0*1im*(-0.5*Ω1*ρ[13,12]*Pz1 - 1196871.01319885*ρ[3,12])
		du[3,13] = -5026548.21497941*ρ[3,13] - 1.0*1im*(-Δ*ρ[3,13] - 0.5*Ω1*ρ[13,13]*Pz1 - 0.353552889862801*Ω1*ρ[3,2]*Px1 + 0.5*Ω1*ρ[3,3]*Pz1 + 0.353553895977235*Ω1*ρ[3,4]*Px1 - 0.205619589808608*Ω1*ρ[3,5]*Px1 + 0.29078878665932*Ω1*ρ[3,6]*Pz1 + 0.205617856697416*Ω1*ρ[3,7]*Px1)
		du[4,1] = -1.0*1im*(-0.353553895977235*Ω1*ρ[13,1]*Px1 + 139719.693466187*ρ[4,1])
		du[4,2] = -1.0*1im*(-0.353553895977235*Ω1*ρ[13,2]*Px1 - 0.353552889862801*Ω1ᶜ*ρ[4,13]*Px1 - 21.4009704589844*ρ[4,2])
		du[4,3] = -1.0*1im*(-0.353553895977235*Ω1*ρ[13,3]*Px1 + 0.5*Ω1ᶜ*ρ[4,13]*Pz1 - 10.7021484375*ρ[4,3])
		du[4,4] = 2504080.39374506*ρ[13,13] - 1.0*1im*(-0.353553895977235*Ω1*ρ[13,4]*Px1 + 0.353553895977235*Ω1ᶜ*ρ[4,13]*Px1)
		du[4,5] = -1.0*1im*(-0.353553895977235*Ω1*ρ[13,5]*Px1 - 0.205619589808608*Ω1ᶜ*ρ[4,13]*Px1 - 1105552.27386475*ρ[4,5])
		du[4,6] = -1.0*1im*(-0.353553895977235*Ω1*ρ[13,6]*Px1 + 0.29078878665932*Ω1ᶜ*ρ[4,13]*Pz1 - 1105552.45256042*ρ[4,6])
		du[4,7] = -1.0*1im*(-0.353553895977235*Ω1*ρ[13,7]*Px1 + 0.205617856697416*Ω1ᶜ*ρ[4,13]*Px1 - 1105552.63163757*ρ[4,7])
		du[4,8] = -1.0*1im*(-0.353553895977235*Ω1*ρ[13,8]*Px1 - 1196922.92208862*ρ[4,8])
		du[4,9] = -1.0*1im*(-0.353553895977235*Ω1*ρ[13,9]*Px1 - 1196912.62124634*ρ[4,9])
		du[4,10] = -1.0*1im*(-0.353553895977235*Ω1*ρ[13,10]*Px1 - 1196902.31977844*ρ[4,10])
		du[4,11] = -1.0*1im*(-0.353553895977235*Ω1*ρ[13,11]*Px1 - 1196892.01783752*ρ[4,11])
		du[4,12] = -1.0*1im*(-0.353553895977235*Ω1*ρ[13,12]*Px1 - 1196881.71534729*ρ[4,12])
		du[4,13] = -5026548.21497941*ρ[4,13] - 1.0*1im*(-Δ*ρ[4,13] - 0.353553895977235*Ω1*ρ[13,13]*Px1 - 0.353552889862801*Ω1*ρ[4,2]*Px1 + 0.5*Ω1*ρ[4,3]*Pz1 + 0.353553895977235*Ω1*ρ[4,4]*Px1 - 0.205619589808608*Ω1*ρ[4,5]*Px1 + 0.29078878665932*Ω1*ρ[4,6]*Pz1 + 0.205617856697416*Ω1*ρ[4,7]*Px1 - 10.7021484375*ρ[4,13])
		du[5,1] = -1.0*1im*(0.205619589808608*Ω1*ρ[13,1]*Px1 + 1245271.96733093*ρ[5,1])
		du[5,2] = -1.0*1im*(0.205619589808608*Ω1*ρ[13,2]*Px1 - 0.353552889862801*Ω1ᶜ*ρ[5,13]*Px1 + 1105530.87289429*ρ[5,2])
		du[5,3] = -1.0*1im*(0.205619589808608*Ω1*ρ[13,3]*Px1 + 0.5*Ω1ᶜ*ρ[5,13]*Pz1 + 1105541.57171631*ρ[5,3])
		du[5,4] = -1.0*1im*(0.205619589808608*Ω1*ρ[13,4]*Px1 + 0.353553895977235*Ω1ᶜ*ρ[5,13]*Px1 + 1105552.27386475*ρ[5,4])
		du[5,5] = 846966.013293444*ρ[13,13] - 1.0*1im*(0.205619589808608*Ω1*ρ[13,5]*Px1 - 0.205619589808608*Ω1ᶜ*ρ[5,13]*Px1)
		du[5,6] = -1.0*1im*(0.205619589808608*Ω1*ρ[13,6]*Px1 + 0.29078878665932*Ω1ᶜ*ρ[5,13]*Pz1 - 0.178695678710938*ρ[5,6])
		du[5,7] = -1.0*1im*(0.205619589808608*Ω1*ρ[13,7]*Px1 + 0.205617856697416*Ω1ᶜ*ρ[5,13]*Px1 - 0.357772827148438*ρ[5,7])
		du[5,8] = -1.0*1im*(0.205619589808608*Ω1*ρ[13,8]*Px1 - 91370.648223877*ρ[5,8])
		du[5,9] = -1.0*1im*(0.205619589808608*Ω1*ρ[13,9]*Px1 - 91360.3473815918*ρ[5,9])
		du[5,10] = -1.0*1im*(0.205619589808608*Ω1*ρ[13,10]*Px1 - 91350.0459136963*ρ[5,10])
		du[5,11] = -1.0*1im*(0.205619589808608*Ω1*ρ[13,11]*Px1 - 91339.7439727783*ρ[5,11])
		du[5,12] = -1.0*1im*(0.205619589808608*Ω1*ρ[13,12]*Px1 - 91329.4414825439*ρ[5,12])
		du[5,13] = -5026548.21497941*ρ[5,13] - 1.0*1im*(-Δ*ρ[5,13] + 0.205619589808608*Ω1*ρ[13,13]*Px1 - 0.353552889862801*Ω1*ρ[5,2]*Px1 + 0.5*Ω1*ρ[5,3]*Pz1 + 0.353553895977235*Ω1*ρ[5,4]*Px1 - 0.205619589808608*Ω1*ρ[5,5]*Px1 + 0.29078878665932*Ω1*ρ[5,6]*Pz1 + 0.205617856697416*Ω1*ρ[5,7]*Px1 + 1105541.57171631*ρ[5,13])
		du[6,1] = -1.0*1im*(-0.29078878665932*Ω1*ρ[13,1]*Pz1 + 1245272.14602661*ρ[6,1])
		du[6,2] = -1.0*1im*(-0.29078878665932*Ω1*ρ[13,2]*Pz1 - 0.353552889862801*Ω1ᶜ*ρ[6,13]*Px1 + 1105531.05158997*ρ[6,2])
		du[6,3] = -1.0*1im*(-0.29078878665932*Ω1*ρ[13,3]*Pz1 + 0.5*Ω1ᶜ*ρ[6,13]*Pz1 + 1105541.75041199*ρ[6,3])
		du[6,4] = -1.0*1im*(-0.29078878665932*Ω1*ρ[13,4]*Pz1 + 0.353553895977235*Ω1ᶜ*ρ[6,13]*Px1 + 1105552.45256042*ρ[6,4])
		du[6,5] = -1.0*1im*(-0.29078878665932*Ω1*ρ[13,5]*Pz1 - 0.205619589808608*Ω1ᶜ*ρ[6,13]*Px1 + 0.178695678710938*ρ[6,5])
		du[6,6] = 846958.884766602*ρ[13,13] - 1.0*1im*(-0.29078878665932*Ω1*ρ[13,6]*Pz1 + 0.29078878665932*Ω1ᶜ*ρ[6,13]*Pz1)
		du[6,7] = -1.0*1im*(-0.29078878665932*Ω1*ρ[13,7]*Pz1 + 0.205617856697416*Ω1ᶜ*ρ[6,13]*Px1 - 0.1790771484375*ρ[6,7])
		du[6,8] = -1.0*1im*(-0.29078878665932*Ω1*ρ[13,8]*Pz1 - 91370.4695281982*ρ[6,8])
		du[6,9] = -1.0*1im*(-0.29078878665932*Ω1*ρ[13,9]*Pz1 - 91360.1686859131*ρ[6,9])
		du[6,10] = -1.0*1im*(-0.29078878665932*Ω1*ρ[13,10]*Pz1 - 91349.8672180176*ρ[6,10])
		du[6,11] = -1.0*1im*(-0.29078878665932*Ω1*ρ[13,11]*Pz1 - 91339.5652770996*ρ[6,11])
		du[6,12] = -1.0*1im*(-0.29078878665932*Ω1*ρ[13,12]*Pz1 - 91329.2627868652*ρ[6,12])
		du[6,13] = -5026548.21497941*ρ[6,13] - 1.0*1im*(-Δ*ρ[6,13] - 0.29078878665932*Ω1*ρ[13,13]*Pz1 - 0.353552889862801*Ω1*ρ[6,2]*Px1 + 0.5*Ω1*ρ[6,3]*Pz1 + 0.353553895977235*Ω1*ρ[6,4]*Px1 - 0.205619589808608*Ω1*ρ[6,5]*Px1 + 0.29078878665932*Ω1*ρ[6,6]*Pz1 + 0.205617856697416*Ω1*ρ[6,7]*Px1 + 1105541.75041199*ρ[6,13])
		du[7,1] = -1.0*1im*(-0.205617856697416*Ω1*ρ[13,1]*Px1 + 1245272.32510376*ρ[7,1])
		du[7,2] = -1.0*1im*(-0.205617856697416*Ω1*ρ[13,2]*Px1 - 0.353552889862801*Ω1ᶜ*ρ[7,13]*Px1 + 1105531.23066711*ρ[7,2])
		du[7,3] = -1.0*1im*(-0.205617856697416*Ω1*ρ[13,3]*Px1 + 0.5*Ω1ᶜ*ρ[7,13]*Pz1 + 1105541.92948914*ρ[7,3])
		du[7,4] = -1.0*1im*(-0.205617856697416*Ω1*ρ[13,4]*Px1 + 0.353553895977235*Ω1ᶜ*ρ[7,13]*Px1 + 1105552.63163757*ρ[7,4])
		du[7,5] = -1.0*1im*(-0.205617856697416*Ω1*ρ[13,5]*Px1 - 0.205619589808608*Ω1ᶜ*ρ[7,13]*Px1 + 0.357772827148438*ρ[7,5])
		du[7,6] = -1.0*1im*(-0.205617856697416*Ω1*ρ[13,6]*Px1 + 0.29078878665932*Ω1ᶜ*ρ[7,13]*Pz1 + 0.1790771484375*ρ[7,6])
		du[7,7] = 846951.761430674*ρ[13,13] - 1.0*1im*(-0.205617856697416*Ω1*ρ[13,7]*Px1 + 0.205617856697416*Ω1ᶜ*ρ[7,13]*Px1)
		du[7,8] = -1.0*1im*(-0.205617856697416*Ω1*ρ[13,8]*Px1 - 91370.2904510498*ρ[7,8])
		du[7,9] = -1.0*1im*(-0.205617856697416*Ω1*ρ[13,9]*Px1 - 91359.9896087646*ρ[7,9])
		du[7,10] = -1.0*1im*(-0.205617856697416*Ω1*ρ[13,10]*Px1 - 91349.6881408691*ρ[7,10])
		du[7,11] = -1.0*1im*(-0.205617856697416*Ω1*ρ[13,11]*Px1 - 91339.3861999512*ρ[7,11])
		du[7,12] = -1.0*1im*(-0.205617856697416*Ω1*ρ[13,12]*Px1 - 91329.0837097168*ρ[7,12])
		du[7,13] = -5026548.21497941*ρ[7,13] - 1.0*1im*(-Δ*ρ[7,13] - 0.205617856697416*Ω1*ρ[13,13]*Px1 - 0.353552889862801*Ω1*ρ[7,2]*Px1 + 0.5*Ω1*ρ[7,3]*Pz1 + 0.353553895977235*Ω1*ρ[7,4]*Px1 - 0.205619589808608*Ω1*ρ[7,5]*Px1 + 0.29078878665932*Ω1*ρ[7,6]*Pz1 + 0.205617856697416*Ω1*ρ[7,7]*Px1 + 1105541.92948914*ρ[7,13])
		du[8,1] = -1336642.61555481*1im*ρ[8,1]
		du[8,2] = -1.0*1im*(-0.353552889862801*Ω1ᶜ*ρ[8,13]*Px1 + 1196901.52111816*ρ[8,2])
		du[8,3] = -1.0*1im*(0.5*Ω1ᶜ*ρ[8,13]*Pz1 + 1196912.21994019*ρ[8,3])
		du[8,4] = -1.0*1im*(0.353553895977235*Ω1ᶜ*ρ[8,13]*Px1 + 1196922.92208862*ρ[8,4])
		du[8,5] = -1.0*1im*(-0.205619589808608*Ω1ᶜ*ρ[8,13]*Px1 + 91370.648223877*ρ[8,5])
		du[8,6] = -1.0*1im*(0.29078878665932*Ω1ᶜ*ρ[8,13]*Pz1 + 91370.4695281982*ρ[8,6])
		du[8,7] = -1.0*1im*(0.205617856697416*Ω1ᶜ*ρ[8,13]*Px1 + 91370.2904510498*ρ[8,7])
		du[8,9] = -10.3008422851563*1im*ρ[8,9]
		du[8,10] = -20.6023101806641*1im*ρ[8,10]
		du[8,11] = -30.9042510986328*1im*ρ[8,11]
		du[8,12] = -41.2067413330078*1im*ρ[8,12]
		du[8,13] = -5026548.21497941*ρ[8,13] - 1.0*1im*(-Δ*ρ[8,13] - 0.353552889862801*Ω1*ρ[8,2]*Px1 + 0.5*Ω1*ρ[8,3]*Pz1 + 0.353553895977235*Ω1*ρ[8,4]*Px1 - 0.205619589808608*Ω1*ρ[8,5]*Px1 + 0.29078878665932*Ω1*ρ[8,6]*Pz1 + 0.205617856697416*Ω1*ρ[8,7]*Px1 + 1196912.21994019*ρ[8,13])
		du[9,1] = -1336632.31471252*1im*ρ[9,1]
		du[9,2] = -1.0*1im*(-0.353552889862801*Ω1ᶜ*ρ[9,13]*Px1 + 1196891.22027588*ρ[9,2])
		du[9,3] = -1.0*1im*(0.5*Ω1ᶜ*ρ[9,13]*Pz1 + 1196901.9190979*ρ[9,3])
		du[9,4] = -1.0*1im*(0.353553895977235*Ω1ᶜ*ρ[9,13]*Px1 + 1196912.62124634*ρ[9,4])
		du[9,5] = -1.0*1im*(-0.205619589808608*Ω1ᶜ*ρ[9,13]*Px1 + 91360.3473815918*ρ[9,5])
		du[9,6] = -1.0*1im*(0.29078878665932*Ω1ᶜ*ρ[9,13]*Pz1 + 91360.1686859131*ρ[9,6])
		du[9,7] = -1.0*1im*(0.205617856697416*Ω1ᶜ*ρ[9,13]*Px1 + 91359.9896087646*ρ[9,7])
		du[9,8] = 10.3008422851563*1im*ρ[9,8]
		du[9,10] = -10.3014678955078*1im*ρ[9,10]
		du[9,11] = -20.6034088134766*1im*ρ[9,11]
		du[9,12] = -30.9058990478516*1im*ρ[9,12]
		du[9,13] = -5026548.21497941*ρ[9,13] - 1.0*1im*(-Δ*ρ[9,13] - 0.353552889862801*Ω1*ρ[9,2]*Px1 + 0.5*Ω1*ρ[9,3]*Pz1 + 0.353553895977235*Ω1*ρ[9,4]*Px1 - 0.205619589808608*Ω1*ρ[9,5]*Px1 + 0.29078878665932*Ω1*ρ[9,6]*Pz1 + 0.205617856697416*Ω1*ρ[9,7]*Px1 + 1196901.9190979*ρ[9,13])
		du[10,1] = -1336622.01324463*1im*ρ[10,1]
		du[10,2] = -1.0*1im*(-0.353552889862801*Ω1ᶜ*ρ[10,13]*Px1 + 1196880.91880798*ρ[10,2])
		du[10,3] = -1.0*1im*(0.5*Ω1ᶜ*ρ[10,13]*Pz1 + 1196891.61763*ρ[10,3])
		du[10,4] = -1.0*1im*(0.353553895977235*Ω1ᶜ*ρ[10,13]*Px1 + 1196902.31977844*ρ[10,4])
		du[10,5] = -1.0*1im*(-0.205619589808608*Ω1ᶜ*ρ[10,13]*Px1 + 91350.0459136963*ρ[10,5])
		du[10,6] = -1.0*1im*(0.29078878665932*Ω1ᶜ*ρ[10,13]*Pz1 + 91349.8672180176*ρ[10,6])
		du[10,7] = -1.0*1im*(0.205617856697416*Ω1ᶜ*ρ[10,13]*Px1 + 91349.6881408691*ρ[10,7])
		du[10,8] = 20.6023101806641*1im*ρ[10,8]
		du[10,9] = 10.3014678955078*1im*ρ[10,9]
		du[10,11] = -10.3019409179688*1im*ρ[10,11]
		du[10,12] = -20.6044311523438*1im*ρ[10,12]
		du[10,13] = -5026548.21497941*ρ[10,13] - 1.0*1im*(-Δ*ρ[10,13] - 0.353552889862801*Ω1*ρ[10,2]*Px1 + 0.5*Ω1*ρ[10,3]*Pz1 + 0.353553895977235*Ω1*ρ[10,4]*Px1 - 0.205619589808608*Ω1*ρ[10,5]*Px1 + 0.29078878665932*Ω1*ρ[10,6]*Pz1 + 0.205617856697416*Ω1*ρ[10,7]*Px1 + 1196891.61763*ρ[10,13])
		du[11,1] = -1336611.71130371*1im*ρ[11,1]
		du[11,2] = -1.0*1im*(-0.353552889862801*Ω1ᶜ*ρ[11,13]*Px1 + 1196870.61686707*ρ[11,2])
		du[11,3] = -1.0*1im*(0.5*Ω1ᶜ*ρ[11,13]*Pz1 + 1196881.31568909*ρ[11,3])
		du[11,4] = -1.0*1im*(0.353553895977235*Ω1ᶜ*ρ[11,13]*Px1 + 1196892.01783752*ρ[11,4])
		du[11,5] = -1.0*1im*(-0.205619589808608*Ω1ᶜ*ρ[11,13]*Px1 + 91339.7439727783*ρ[11,5])
		du[11,6] = -1.0*1im*(0.29078878665932*Ω1ᶜ*ρ[11,13]*Pz1 + 91339.5652770996*ρ[11,6])
		du[11,7] = -1.0*1im*(0.205617856697416*Ω1ᶜ*ρ[11,13]*Px1 + 91339.3861999512*ρ[11,7])
		du[11,8] = 30.9042510986328*1im*ρ[11,8]
		du[11,9] = 20.6034088134766*1im*ρ[11,9]
		du[11,10] = 10.3019409179688*1im*ρ[11,10]
		du[11,12] = -10.302490234375*1im*ρ[11,12]
		du[11,13] = -5026548.21497941*ρ[11,13] - 1.0*1im*(-Δ*ρ[11,13] - 0.353552889862801*Ω1*ρ[11,2]*Px1 + 0.5*Ω1*ρ[11,3]*Pz1 + 0.353553895977235*Ω1*ρ[11,4]*Px1 - 0.205619589808608*Ω1*ρ[11,5]*Px1 + 0.29078878665932*Ω1*ρ[11,6]*Pz1 + 0.205617856697416*Ω1*ρ[11,7]*Px1 + 1196881.31568909*ρ[11,13])
		du[12,1] = -1336601.40881348*1im*ρ[12,1]
		du[12,2] = -1.0*1im*(-0.353552889862801*Ω1ᶜ*ρ[12,13]*Px1 + 1196860.31437683*ρ[12,2])
		du[12,3] = -1.0*1im*(0.5*Ω1ᶜ*ρ[12,13]*Pz1 + 1196871.01319885*ρ[12,3])
		du[12,4] = -1.0*1im*(0.353553895977235*Ω1ᶜ*ρ[12,13]*Px1 + 1196881.71534729*ρ[12,4])
		du[12,5] = -1.0*1im*(-0.205619589808608*Ω1ᶜ*ρ[12,13]*Px1 + 91329.4414825439*ρ[12,5])
		du[12,6] = -1.0*1im*(0.29078878665932*Ω1ᶜ*ρ[12,13]*Pz1 + 91329.2627868652*ρ[12,6])
		du[12,7] = -1.0*1im*(0.205617856697416*Ω1ᶜ*ρ[12,13]*Px1 + 91329.0837097168*ρ[12,7])
		du[12,8] = 41.2067413330078*1im*ρ[12,8]
		du[12,9] = 30.9058990478516*1im*ρ[12,9]
		du[12,10] = 20.6044311523438*1im*ρ[12,10]
		du[12,11] = 10.302490234375*1im*ρ[12,11]
		du[12,13] = -5026548.21497941*ρ[12,13] - 1.0*1im*(-Δ*ρ[12,13] - 0.353552889862801*Ω1*ρ[12,2]*Px1 + 0.5*Ω1*ρ[12,3]*Pz1 + 0.353553895977235*Ω1*ρ[12,4]*Px1 - 0.205619589808608*Ω1*ρ[12,5]*Px1 + 0.29078878665932*Ω1*ρ[12,6]*Pz1 + 0.205617856697416*Ω1*ρ[12,7]*Px1 + 1196871.01319885*ρ[12,13])
		du[13,1] = -5026548.21497941*ρ[13,1] - 1.0*1im*(Δ*ρ[13,1] + 0.353552889862801*Ω1ᶜ*ρ[2,1]*Px1 - 0.5*Ω1ᶜ*ρ[3,1]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,1]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,1]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,1]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,1]*Px1 + 139730.395614624*ρ[13,1])
		du[13,2] = -5026548.21497941*ρ[13,2] - 1.0*1im*(Δ*ρ[13,2] + 0.353552889862801*Ω1ᶜ*ρ[2,2]*Px1 - 0.353552889862801*Ω1ᶜ*ρ[13,13]*Px1 - 0.5*Ω1ᶜ*ρ[3,2]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,2]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,2]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,2]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,2]*Px1 - 10.6988220214844*ρ[13,2])
		du[13,3] = -5026548.21497941*ρ[13,3] - 1.0*1im*(Δ*ρ[13,3] + 0.353552889862801*Ω1ᶜ*ρ[2,3]*Px1 + 0.5*Ω1ᶜ*ρ[13,13]*Pz1 - 0.5*Ω1ᶜ*ρ[3,3]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,3]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,3]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,3]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,3]*Px1)
		du[13,4] = -5026548.21497941*ρ[13,4] - 1.0*1im*(Δ*ρ[13,4] + 0.353552889862801*Ω1ᶜ*ρ[2,4]*Px1 + 0.353553895977235*Ω1ᶜ*ρ[13,13]*Px1 - 0.5*Ω1ᶜ*ρ[3,4]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,4]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,4]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,4]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,4]*Px1 + 10.7021484375*ρ[13,4])
		du[13,5] = -5026548.21497941*ρ[13,5] - 1.0*1im*(Δ*ρ[13,5] + 0.353552889862801*Ω1ᶜ*ρ[2,5]*Px1 - 0.205619589808608*Ω1ᶜ*ρ[13,13]*Px1 - 0.5*Ω1ᶜ*ρ[3,5]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,5]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,5]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,5]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,5]*Px1 - 1105541.57171631*ρ[13,5])
		du[13,6] = -5026548.21497941*ρ[13,6] - 1.0*1im*(Δ*ρ[13,6] + 0.353552889862801*Ω1ᶜ*ρ[2,6]*Px1 + 0.29078878665932*Ω1ᶜ*ρ[13,13]*Pz1 - 0.5*Ω1ᶜ*ρ[3,6]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,6]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,6]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,6]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,6]*Px1 - 1105541.75041199*ρ[13,6])
		du[13,7] = -5026548.21497941*ρ[13,7] - 1.0*1im*(Δ*ρ[13,7] + 0.353552889862801*Ω1ᶜ*ρ[2,7]*Px1 + 0.205617856697416*Ω1ᶜ*ρ[13,13]*Px1 - 0.5*Ω1ᶜ*ρ[3,7]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,7]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,7]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,7]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,7]*Px1 - 1105541.92948914*ρ[13,7])
		du[13,8] = -5026548.21497941*ρ[13,8] - 1.0*1im*(Δ*ρ[13,8] + 0.353552889862801*Ω1ᶜ*ρ[2,8]*Px1 - 0.5*Ω1ᶜ*ρ[3,8]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,8]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,8]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,8]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,8]*Px1 - 1196912.21994019*ρ[13,8])
		du[13,9] = -5026548.21497941*ρ[13,9] - 1.0*1im*(Δ*ρ[13,9] + 0.353552889862801*Ω1ᶜ*ρ[2,9]*Px1 - 0.5*Ω1ᶜ*ρ[3,9]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,9]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,9]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,9]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,9]*Px1 - 1196901.9190979*ρ[13,9])
		du[13,10] = -5026548.21497941*ρ[13,10] - 1.0*1im*(Δ*ρ[13,10] + 0.353552889862801*Ω1ᶜ*ρ[2,10]*Px1 - 0.5*Ω1ᶜ*ρ[3,10]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,10]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,10]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,10]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,10]*Px1 - 1196891.61763*ρ[13,10])
		du[13,11] = -5026548.21497941*ρ[13,11] - 1.0*1im*(Δ*ρ[13,11] + 0.353552889862801*Ω1ᶜ*ρ[2,11]*Px1 - 0.5*Ω1ᶜ*ρ[3,11]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,11]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,11]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,11]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,11]*Px1 - 1196881.31568909*ρ[13,11])
		du[13,12] = -5026548.21497941*ρ[13,12] - 1.0*1im*(Δ*ρ[13,12] + 0.353552889862801*Ω1ᶜ*ρ[2,12]*Px1 - 0.5*Ω1ᶜ*ρ[3,12]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,12]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,12]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,12]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,12]*Px1 - 1196871.01319885*ρ[13,12])
		du[13,13] = -10053096.4299588*ρ[13,13] - 1.0*1im*(-0.353552889862801*Ω1*ρ[13,2]*Px1 + 0.5*Ω1*ρ[13,3]*Pz1 + 0.353553895977235*Ω1*ρ[13,4]*Px1 - 0.205619589808608*Ω1*ρ[13,5]*Px1 + 0.29078878665932*Ω1*ρ[13,6]*Pz1 + 0.205617856697416*Ω1*ρ[13,7]*Px1 + 0.353552889862801*Ω1ᶜ*ρ[2,13]*Px1 - 0.5*Ω1ᶜ*ρ[3,13]*Pz1 - 0.353553895977235*Ω1ᶜ*ρ[4,13]*Px1 + 0.205619589808608*Ω1ᶜ*ρ[5,13]*Px1 - 0.29078878665932*Ω1ᶜ*ρ[6,13]*Pz1 - 0.205617856697416*Ω1ᶜ*ρ[7,13]*Px1)
	 end 
 	 nothing 
 end