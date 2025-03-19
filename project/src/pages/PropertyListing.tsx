import React from 'react';
import { useParams } from 'react-router-dom';
import { Building2, Bed, Bath, Square, Calendar } from 'lucide-react';
// import { getPropertyById } from '../lib/supabase';
// import { Property } from '../lib/supabase';

type Property = {
  id: string;
  title: string;
  price: number;
  bedrooms: number;
  bathrooms: number;
  square_feet: number;
  year_built: number;
  description: string;
  features: string[];
  images: { url: string }[];
  address: string;
  city: string;
  state: string;
  zip_code: string;
};

export function PropertyListing() {
  const { id } = useParams<{ id: string }>();
  const [property] = React.useState<Property | null>(null);
  const [loading] = React.useState(true);

  React.useEffect(() => {
    if (id) {
      // getPropertyById(id)
      //   .then(setProperty)
      //   .finally(() => setLoading(false));
    }
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!property) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <h1 className="text-2xl text-gray-600">Property not found</h1>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Image Gallery */}
        <div className="space-y-4">
          <div className="aspect-w-16 aspect-h-9 rounded-lg overflow-hidden">
            <img
              src={property.images[0]?.url || 'https://images.unsplash.com/photo-1564013799919-ab600027ffc6'}
              alt={property.title}
              className="object-cover w-full h-full"
            />
          </div>
          <div className="grid grid-cols-4 gap-4">
            {property.images.slice(1).map((image: { url: string }, index: number) => (
              <div key={index} className="aspect-w-1 aspect-h-1 rounded-lg overflow-hidden">
                <img src={image.url} alt={`${property.title} - ${index + 2}`} className="object-cover" />
              </div>
            ))}
          </div>
        </div>

        {/* Property Details */}
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">{property.title}</h1>
            <p className="text-2xl text-blue-600 font-semibold mt-2">
              ${property.price.toLocaleString()}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center space-x-2">
              <Bed className="h-5 w-5 text-gray-600" />
              <span>{property.bedrooms} Bedrooms</span>
            </div>
            <div className="flex items-center space-x-2">
              <Bath className="h-5 w-5 text-gray-600" />
              <span>{property.bathrooms} Bathrooms</span>
            </div>
            <div className="flex items-center space-x-2">
              <Square className="h-5 w-5 text-gray-600" />
              <span>{property.square_feet.toLocaleString()} sq ft</span>
            </div>
            <div className="flex items-center space-x-2">
              <Calendar className="h-5 w-5 text-gray-600" />
              <span>Built {property.year_built}</span>
            </div>
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-2">Description</h2>
            <p className="text-gray-600">{property.description}</p>
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-2">Features</h2>
            <ul className="grid grid-cols-2 gap-2">
              {property.features.map((feature: string, index: number) => (
                <li key={index} className="flex items-center space-x-2">
                  <Building2 className="h-4 w-4 text-blue-600" />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-2">Location</h2>
            <p className="text-gray-600">
              {property.address}, {property.city}, {property.state} {property.zip_code}
            </p>
          </div>

          <button className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition">
            Contact Agent
          </button>
        </div>
      </div>
    </div>
  );
}