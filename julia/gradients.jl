# Hinge:
function gradient(x,w,y)
    z = ifelse(y==1,1,-1)
    if z.*w'x <= 1
        𝐠 = -z .* w
    else
        𝐠 = zeros(length(w))
    end
    return 𝐠
end;

# MSE:
function gradient(x,w,y)
    𝐠 = 2 * 𝛔(w'x) * (1-𝛔(w'x)) * (𝛔(w'x) - y) .* w
    return 𝐠
end;

# Cross-entropy:
function gradient(x,w,y)
    𝐠 = (𝛔(w'x) - y) .* w
    return 𝐠
end;