#
#
#           Gabriel Vinicius Ferreira.
#           Sérgio Roberto Lopes, Thiago de Lima Prado.
#
#       I want to see how the accuracy of a neural network (more specifically, an MMLP) 
#   is affected when I train it using small values of the minimum recurrence threshold while
#   applying it to the classification of a Beta-X system.
#
#   Note: The Beta-X is commonly referred to as "Bernouli-shift generalized"
#   modelo ISO-4 para referencias
# ==============================================================================================
#               LIBRARIES
using Microstates
using ProgressMeter

using JLD2
using Flux
using Random
using Statistics
using LinearAlgebra
# ==============================================================================================
#               CONFIG THE RANDOM
rng = MersenneTwister()
Random.seed!()
# ==============================================================================================
#               SETTINGS
#       - Values of β
const β = [2.59, 2.99, 3.59, 3.99, 4.59, 4.99]
#       - Number os elements of beta-x
const timesize = 1000
#       - Values of minimum threshold
const min_thresholds = range(0.0, 0.01, 10)
#       - Resolution for the maximum threshold
const max_threshold_length = 10
#       - Epochs of train...
const epochs = 4
#       - Size of microstates...
const n = 3
#       - Power vector =D
const vect = power_vector(n)
#       - Learning rate
const learning_rate = 0.0001
# ==============================================================================================
#       Calculates the beta x serie from group of initial values.
function beta_x(x; transient = round(Int, (10 * timesize)))
    serie = zeros(Float64, (1, timesize, length(x), length(β)))
    for b_index in eachindex(β)
        for x_index in eachindex(x)
            before = x[x_index]

            for time = 1:(timesize + transient)
                after = before * β[b_index]
                while(after > 1.0)
                    after = after - 1.0
                end

                before = after

                if (time > transient)
                    serie[1, time-transient, x_index, b_index] = before
                end
            end
        end
    end

    return serie
end
# ==============================================================================================
#           Creates a new process.
function create_process()
    xo_to_entropy = range(0.00001, 0.99999, 10)
    xo_to_train = rand(Float64, 1200)
    xo_to_test = rand(Float64, 300)

    #           Checks if the test values are different from the training values.
    for i in eachindex(xo_to_test)
        while (xo_to_test[i] in xo_to_train)
            new_value = rand(Float64, 1)
            while (new_value in xo_to_test)
                new_value = rand(Float64, 1)
            end
            xo_to_test[i] = new_value
        end
    end

    serie_to_entropy = beta_x(xo_to_entropy)
    serie_to_train = beta_x(xo_to_train)
    serie_to_test = beta_x(xo_to_test)

    save_object("near-beta-x/data/serie-train.dat", serie_to_train)
    save_object("near-beta-x/data/serie-test.dat", serie_to_test)

    accuracy = zeros(Float64, (length(min_thresholds), max_threshold_length, epochs))
    loss = zeros(Float64, (length(min_thresholds), max_threshold_length, epochs))
    entropy = zeros(Float64, max_threshold_length)
    max_thresholds = range(0.0, 1.0, max_threshold_length)

    for thres_max in eachindex(max_thresholds)
        s = []
        for samp in 1:length(xo_to_entropy)
            probs, _ = microstates(serie_to_entropy[:, :, samp, 1], max_thresholds[thres_max], n; samples_percent = 0.1, vect = vect)
            push!(s, Microstates.entropy(probs))
        end
        entropy[thres_max] = mean(s)
    end

    max_std_etr = findmax(entropy)[2]
    etr_threshold_reference = (max_thresholds[max_std_etr - floor(Int, max_threshold_length / 10)], max_thresholds[max_std_etr + floor(Int, max_threshold_length / 10)])

    max_thresholds = range(etr_threshold_reference[1], etr_threshold_reference[2], max_threshold_length)

    entropy = zeros(Float64, length(min_thresholds), length(max_thresholds))

    for thres_min in eachindex(min_thresholds)
        for thres_max in eachindex(max_thresholds)
            s = []
            for samp in 1:length(xo_to_entropy)
                probs, _ = microstates(serie_to_entropy[:, :, samp, 1], (min_thresholds[thres_min], max_thresholds[thres_max]), n; samples_percent = 0.1, vect = vect, recurr = crd_recurrence)
                push!(s, Microstates.entropy(probs))
            end
            entropy[thres_min, thres_max] = mean(s)
        end
    end

    model = Chain(
        Dense(2^(n * n) => 128, identity),
        Dense(128 => 64, selu),
        Dense(64 => 32, selu),
        Dense(32 => length(β)),
        softmax
    )

    model = f64(model)

    save_object("near-beta-x/data/loss.dat", loss)
    save_object("near-beta-x/data/accuracy.dat", accuracy)
    save_object("near-beta-x/data/entropy.dat", entropy)
    save_object("near-beta-x/data/max-threshold.dat", max_thresholds)
    save_object("near-beta-x/data/model.net", model)
    save_object("near-beta-x/data/status.dat", [1, 1])
end
# ==============================================================================================
function calc_accuracy(predict, trusty)
    conf = zeros(Int, length(β), length(β))
    sz = size(predict, 2)

    for i in 1:sz
        mx_prd = findmax(predict[:, i])
        mx_trt = findmax(trusty[:, i])
        conf[mx_prd[2], mx_trt[2]] += 1
    end

    return tr(conf) / sum(conf)
end
# ==============================================================================================
#           Entry point of the script.
function main()
    #           Create a new process =D
    if(!isfile("near-beta-x/data/status.dat"))
        create_process()
    end
    #           Load the process...
    loss = load_object("near-beta-x/data/loss.dat")
    accuracy = load_object("near-beta-x/data/accuracy.dat")
    max_thresholds = load_object("near-beta-x/data/max-threshold.dat")
    serie_to_test = load_object("near-beta-x/data/serie-test.dat")
    serie_to_train = load_object("near-beta-x/data/serie-train.dat")
    status = load_object("near-beta-x/data/status.dat")

    #           Make our labels
    train_labels = ones(Float64, size(serie_to_train, 3), length(β))
    test_labels = ones(Float64, size(serie_to_test, 3), length(β))

    for b_index in eachindex(β)
        train_labels[:, b_index] .*= β[b_index]
        test_labels[:, b_index] .*= β[b_index]
    end

    train_labels = reshape(train_labels, size(serie_to_train, 3) * length(β))
    test_labels = reshape(test_labels, size(serie_to_test, 3) * length(β))

    train_labels = Flux.onehotbatch(train_labels, β)
    test_labels = Flux.onehotbatch(test_labels, β)

    #           Do it!!!
    for min_thres in status[1]:length(min_thresholds)
        for max_thres in status[2]:length(max_thresholds)

            probs_to_train = zeros(Float64, 2^(n * n), size(serie_to_train, 3), length(β))
            probs_to_test = zeros(Float64, 2^(n * n), size(serie_to_test, 3), length(β))

            for b_index in eachindex(β)
                Threads.@threads for samp in 1:size(serie_to_train, 3)
                    probs_to_train[:, samp, b_index] .= microstates(serie_to_train[:, :, samp, b_index], (min_thresholds[min_thres], max_thresholds[max_thres]), n; samples_percent = 0.1, vect = vect, recurr = crd_recurrence)[1]
                    if (samp <= size(serie_to_test, 3))
                        probs_to_test[:, samp, b_index] .= microstates(serie_to_test[:, :, samp, b_index], (min_thresholds[min_thres], max_thresholds[max_thres]), n; samples_percent = 0.1, vect = vect, recurr = crd_recurrence)[1]
                    end
                end
            end

            model = load_object("near-beta-x/data/model.net")
            loader = Flux.DataLoader((reshape(probs_to_train, 2^(n * n), size(serie_to_train, 3) * length(β)), train_labels), batchsize = 32, shuffle = true)
            opt = Flux.setup(Flux.Adam(learning_rate), model)

            for epc in 1:epochs
                losses = []
                for (x, y) in loader
                    ld_loss, grads = Flux.withgradient(model) do m
                        y_hat = m(x)
                        Flux.logitcrossentropy(y_hat, y)
                    end
                    push!(losses, ld_loss)
                    Flux.update!(opt, model, grads[1])
                end
                loss[min_thres, max_thres, epc] = mean(losses)
                accuracy[min_thres, max_thres, epc] = calc_accuracy(model(reshape(probs_to_test, 2^(n * n), size(serie_to_test, 3) * length(β))), test_labels)
            end

            save_object("near-beta-x/data/accuracy.dat", accuracy)
            save_object("near-beta-x/data/loss.dat", loss)
            save_object("near-beta-x/data/status.dat", [min_thres, max_thres])
        end
    end
end
# ==============================================================================================
main()