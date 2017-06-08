/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <float.h>
#include "particle_filter.h"

using namespace std;

#define NUM_PARTICLES (100)

void ParticleFilter::init(double gps_x, double gps_y, double gps_theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  default_random_engine gen;
  double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  normal_distribution<double> dist_x(gps_x, std_x);
  normal_distribution<double> dist_y(gps_y, std_y);
  normal_distribution<double> dist_theta(gps_theta, std_theta);

  num_particles = NUM_PARTICLES;

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    double sample_x, sample_y, sample_theta;

    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    p.id = i;
    p.x = sample_x;
    p.y = sample_y;
    p.theta = sample_theta;
    p.weight = 1.0;

    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();

    particles.push_back(p);
    weights.push_back(1.0);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

  std_x = std_pos[0];
  std_y = std_pos[1];
  std_theta = std_pos[2];

  for (int i = 0; i < num_particles; i++) {
      double x0, y0, theta0;
      double x1, y1, theta1;

      x0 = particles[i].x;
      y0 = particles[i].y;
      theta0 = particles[i].theta;

      if(yaw_rate == 0) {
         x1 = x0 + velocity * delta_t * cos(theta0);
         y1 = y0 + velocity * delta_t * sin(theta0);
         theta1 = theta0;
      }
      else {
         theta1 = theta0 + (yaw_rate * delta_t);
         x1 = x0 + ((velocity/yaw_rate) * (sin(theta1) - sin(theta0)));
         y1 = y0 + ((velocity/yaw_rate) * (cos(theta0) - cos(theta1)));
      }

      normal_distribution<double> dist_x(x1, std_x);
      normal_distribution<double> dist_y(y1, std_y);
      normal_distribution<double> dist_theta(theta1, std_theta);

      particles[i].x = dist_x(gen);
      particles[i].y = dist_y(gen);
      particles[i].theta = dist_theta(gen);
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  weights.clear();

  for (int i = 0; i < num_particles; i++) {

    double center_x = particles[i].x;
    double center_y = particles[i].y;
    double theta    = particles[i].theta;

    std::vector<LandmarkObs> predictions;
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
      
    predictions.clear();
    associations.clear();
    sense_x.clear();
    sense_y.clear();

    //Transform observations, rotation first followed by translation
    for(int j = 0; j < observations.size(); j++) {
      LandmarkObs new_obs;

      double obs_x = observations[j].x;
      double obs_y = observations[j].y;
      new_obs.id   = observations[j].id;

      new_obs.x = (obs_x * cos(theta)) - (obs_y * sin(theta)) + center_x;
      new_obs.y = (obs_x * sin(theta)) + (obs_y * cos(theta)) + center_y;

      predictions.push_back(new_obs);
    }


    //Associate with landmark positions
    for(int j = 0; j < predictions.size(); j++) {

      double pos_x = predictions[j].x;
      double pos_y = predictions[j].y;

      double minDist = std::numeric_limits<float>::max();
      int mark_id = -1;
      double mark_x = 0.0;
      double mark_y = 0.0;

      for(int k = 0; k < map_landmarks.landmark_list.size(); k++) {

        double land_x = (double)map_landmarks.landmark_list[k].x_f;
        double land_y = (double)map_landmarks.landmark_list[k].y_f;
          
        double mark_dist = dist(pos_x, pos_y, land_x, land_y);

        if(mark_dist < minDist) {
          minDist = mark_dist;
          mark_id = map_landmarks.landmark_list[k].id_i;
          mark_x  = land_x;
          mark_y  = land_y;
        }
      }

      sense_x.push_back(mark_x);
      sense_y.push_back(mark_y);
      associations.push_back(mark_id);
    }

    SetAssociations(particles[i], associations, sense_x, sense_y);

    //Compute weights using multi-variate probability
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double totalWeight = 1.0;

    for(int j = 0; j < associations.size(); j++) {

      double mark_x = sense_x[j];
      double mark_y = sense_y[j];
      int mark_id   = associations[j];

      double pos_x = predictions[j].x;
      double pos_y = predictions[j].y;

      double val1 = 1 / (2 * M_PI * sig_x * sig_y);
      double val2 = ((pos_x - mark_x) * (pos_x - mark_x)) / (2 * sig_x * sig_x);
      double val3 = ((pos_y - mark_y) * (pos_y - mark_y)) / (2 * sig_y * sig_y);
      double weight = (val1 * exp(-(val2 + val3)));          
       
      if(weight > 0) {
        totalWeight *= weight;
      }
    }

    particles[i].weight = totalWeight;
    weights.push_back(totalWeight);
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine gen;
  Particle p;

  std::vector<Particle> new_particles;
  std::discrete_distribution<> d(this->weights.begin(), this->weights.end());

  new_particles.clear();
  particles.clear();

  for(int i = 0; i < num_particles; i++) {
    int p_idx = d(gen);
    new_particles.push_back(particles[p_idx]);
  } 

  particles = new_particles;
  
  //cout << "Resample Done!" << endl;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

