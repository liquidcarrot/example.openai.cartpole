'use strict'

/**
* This is an example of how to train a Liquid Carrot AI to balance
* a pole using OpenAI's CartPole-v0 gym.
*/

let dotenv = require("dotenv").config()
let async = require('async')
let request = require('request')
let carrot = require('liquid-carrot')(process.env.API_KEY || "API_KEY", process.env.HOST || null)

async.auto({
  // Create OpenAI Environment
  "environment": function(callback) {
    request.post("http://sdcxnbfzarkc.liquidcarrot.io:5000/v1/envs/", {
      json: { "env_id": "CartPole-v0" }
    }, function(error, response, body) {
      callback(error, body)
    })
  },
  // Get Input Layer Size
  "input_layer": ["environment", function(results, callback) {
    request.get("http://sdcxnbfzarkc.liquidcarrot.io:5000/v1/envs/" + results.environment.instance_id + "/observation_space", function(error, response, body) {
      callback(error, JSON.parse(body).info.shape)
    })
  }],
  // Get Output Layer Size
  "output_layer": ["environment", function(results, callback) {
    request.get("http://sdcxnbfzarkc.liquidcarrot.io:5000/v1/envs/" + results.environment.instance_id + "/action_space", function(error, response, body) {
      console.log(JSON.parse(body).info)
      callback(error, JSON.parse(body).info.n)
    })
  }],
  // Create Liquid Carrot Agent
  "agent": ["input_layer", "output_layer", function(results, callback) {
    carrot.agents.create({
      "inputs": results.input_layer[0], // Because results.input_layer.shape exists
      "outputs": 1 // Because results.output_layer.info.name === "Discrete"
    }, callback)
  }],
  // Begin CartPole-v0 Game
  "start": ["environment", function(results, callback) {
    request.post("http://sdcxnbfzarkc.liquidcarrot.io:5000/v1/envs/" + results.environment.instance_id + "/reset/", function(error, response, body) {
      callback(null, JSON.parse(body).observation)
    })
  }],
  // Play CartPole-v0
  "loop": ["start", "agent", function(results, callback) {
    let done = false
    let agent = results.agent
    let environment = results.environment.instance_id
    let observation = results.start
    
    async.whilst(
      () => !done,
      function(callback) {
        async.auto({
          // Make Decision
          "activate": function(callback) {
            carrot.agents.activate(agent.id, { inputs: observation }, callback)
          },
          // Move Cart
          "step": ["activate", function(results, callback) {
            request.post("http://sdcxnbfzarkc.liquidcarrot.io:5000/v1/envs/" + environment + "/step/", {
              json: {
                "instance_id": environment,
                "action": Math.round(results.activate)
              }
            }, function(error, response, body) {
              observation = body.observation, done = body.done
              callback(error, body)
            })
          }],
          // Learn
          "learn": ["step", function(results, callback) {
            carrot.agents.teach(agent.id, { critiques: [1 / results.step.reward] }, callback)
          }]
        }, callback)
      },
      callback
    )
  }]
}, function(error, results) {
  if(error) console.log(error)
  else console.log(results)
})