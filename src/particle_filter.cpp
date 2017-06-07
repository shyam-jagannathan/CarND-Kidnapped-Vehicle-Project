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
  Particle p[NUM_PARTICLES];

  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  normal_distribution<double> dist_x(gps_x, std_x);
  normal_distribution<double> dist_y(gps_y, std_y);
  normal_distribution<double> dist_theta(gps_theta, std_theta);

  this->num_particles = NUM_PARTICLES;

  for (int i = 0; i < this->num_particles; i++) {
    double sample_x, sample_y, sample_theta;

    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    p[i].id = i;
    p[i].x = sample_x;
    p[i].y = sample_y;
    p[i].theta = sample_theta;
    p[i].weight = 1.0;
    
    p[i].associations.clear();
    p[i].sense_x.clear();
    p[i].sense_y.clear();

    this->particles.push_back(p[i]);

//    cout << "Particle initialized: " << p[i].x << ", " << p[i].y << ", " << p[i].theta << endl;
  }

  this->is_initialized = true;

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

  for (int i = 0; i < this->num_particles; i++) {
      double x0, y0, theta0;
      double x1, y1, theta1;

      x0 = this->particles[i].x;
      y0 = this->particles[i].y;
      theta0 = this->particles[i].theta;

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

      this->particles[i].x = dist_x(gen);
      this->particles[i].y = dist_y(gen);
      this->particles[i].theta = dist_theta(gen);

//      cout << "obsrv: " << this->particles[i].x << ", " << this->particles[i].y << "; ";
  }

}

void transformObservations(ParticleFilter *pf, std::vector<LandmarkObs>& observations) {
  //This function performs transformation of observations from particle point of view.
  //The transfomation involves rotation followed by translation

  for (int i = 0; i < pf->particles.size(); i++) {
    double center_x = pf->particles[i].x;
    double center_y = pf->particles[i].y;
    double theta    = pf->particles[i].theta;

    for(int j = 0; j < observations.size(); j++) {
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;

//      double sense_x = (x_center * cos(theta)) - (y_center * sin(theta)) + obs_x;
//      double sense_y = (x_center * sin(theta)) + (y_center * cos(theta)) + obs_y;

      double sense_x = (obs_x * cos(theta)) - (obs_y * sin(theta)) + center_x;
      double sense_y = (obs_x * sin(theta)) + (obs_y * cos(theta)) + center_y;

      pf->particles[i].sense_x.push_back(sense_x);
      pf->particles[i].sense_y.push_back(sense_y);

//      cout << "center: " << center_x << ", " << center_y << ", " << theta << "; ";
//      cout << "obs: " << obs_x << ", " << obs_y << "; ";
//      cout << "sense: " << sense_x << ", " << sense_y << "; ";

    }
  }

}

void findNearestNeighbor(ParticleFilter *pf, Map map_landmarks) {

  for (int i = 0; i < pf->particles.size(); i++) {

    for(int j = 0; j < pf->particles[i].sense_x.size(); j++) {

      double pos_x = pf->particles[i].sense_x[j];
      double pos_y = pf->particles[i].sense_y[j];

      double minDist = std::numeric_limits<float>::max();
      int mark_id = -1;

      for(int k = 0; k < map_landmarks.landmark_list.size(); k++) {

        double land_mark_x = (double)map_landmarks.landmark_list[k].x_f;
        double land_mark_y = (double)map_landmarks.landmark_list[k].y_f;
          
        double mark_dist = dist(pos_x, pos_y, land_mark_x, land_mark_y);

        if(mark_dist < minDist) {
          minDist = mark_dist;
          mark_id = map_landmarks.landmark_list[k].id_i;
        }

      }

      pf->particles[i].associations.push_back(mark_id);

    }
  }
}

void findParticleWeight(ParticleFilter *pf, Map map_landmarks, double std_landmark[]) {

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];

  for (int i = 0; i < pf->particles.size(); ++i) {

    double weight = 1.0;

    for(int j = 0; j < pf->particles[i].associations.size(); j++) {

      double pos_x = pf->particles[i].sense_x[j];
      double pos_y = pf->particles[i].sense_y[j];
      int mark_id = pf->particles[i].associations[j];

      for(int k = 0; k < map_landmarks.landmark_list.size(); k++) {

        if(map_landmarks.landmark_list[k].id_i == mark_id) {

          double land_mark_x = (double)map_landmarks.landmark_list[k].x_f;
          double land_mark_y = (double)map_landmarks.landmark_list[k].y_f;

          double val1 = 1 / (2 * M_PI * sig_x * sig_y);
          double val2 = ((pos_x - land_mark_x) * (pos_x - land_mark_x)) / (2 * sig_x * sig_x);
          double val3 = ((pos_y - land_mark_y) * (pos_y - land_mark_y)) / (2 * sig_y * sig_y);
        
          weight *= (val1 * exp(-(val2 + val3)));

          break;
        }
      }
    }
    //cout << "Particle: " << i << " weight = " << weight << endl;
    pf->particles[i].weight = weight;
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

  transformObservations(this, observations);

  findNearestNeighbor(this, map_landmarks);

  findParticleWeight(this, map_landmarks, std_landmark);

  for(int i = 0; i < this->particles.size(); i++) {
    
    this->weights.push_back(this->particles[i].weight);
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

  for(int j = 0; j < this->particles.size(); j++) {
    this->particles[j].associations.clear();
    this->particles[j].sense_x.clear();
    this->particles[j].sense_y.clear();
  }

  for(int i = 0; i < this->num_particles; i++) {
    int p_idx = d(gen);

    for(int j = 0; j < this->particles.size(); j++) {
      if(this->particles[j].id == p_idx) {
        new_particles.push_back(this->particles[j]);
        break;
      }
    }
  } 

  this->weights.clear();
  this->particles.clear();
  this->particles = new_particles;
  
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

