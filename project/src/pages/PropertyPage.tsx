import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { propertyApi } from "../api/propertyApi";
import { Bed, Bath, Square, MapPin, Home, DollarSign } from "lucide-react";
import { PropertyList } from "../components/PropertyList";
import { EnquiryForm } from "../components/EnquiryForm";

type Property = {
  id: number;
  title: string;
  location: string;
  price: number;
  total_area: number;
  bedrooms: number;
  bathrooms: number;
  images: { url: string }[];
  description: string;
  city: string;
  state: string;
};

export function PropertyPage() {
  const { id } = useParams<{ id: string }>();
  const [property, setProperty] = useState<Property | null>(null);
  const [description, setDescription] = useState("");
  const [recommendations, setRecommendations] = useState<Property[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      if (!id) return;

      setLoading(true);
      const propertyData = await propertyApi.getPropertyDetails(parseInt(id));
      setProperty(propertyData);

      const aiDescription = await propertyApi.generateDescription(propertyData.title);
      setDescription(aiDescription.generated_text);

      const recommendedProps = await propertyApi.getRecommendations(parseInt(id));
      setRecommendations(recommendedProps);
      setLoading(false);
    }

    fetchData();
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!property) {
    return <p className="text-center text-gray-600">Property not found.</p>;
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-16">
      {/* 🔹 Property Images */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <img
          src={property.images[0]?.url || "/assets/images/default-property.jpg"}
          alt={property.title}
          className="w-full h-[400px] object-cover rounded-lg shadow-md"
        />
        <div className="space-y-6">
          <h1 className="text-4xl font-bold text-gray-900">{property.title}</h1>
          <p className="text-gray-500 flex items-center">
            <MapPin className="h-5 w-5 text-blue-600 mr-2" />
            {property.city}, {property.state}
          </p>

          <p className="text-3xl font-semibold text-blue-600 flex items-center">
            <DollarSign className="h-6 w-6 mr-2" />
            ${property.price.toLocaleString()}
          </p>

          {/* 🔹 Property Details */}
          <div className="flex items-center space-x-6 text-gray-600 text-lg mt-4">
            <div className="flex items-center">
              <Bed className="h-6 w-6 mr-2" />
              <span>{property.bedrooms} Beds</span>
            </div>
            <div className="flex items-center">
              <Bath className="h-6 w-6 mr-2" />
              <span>{property.bathrooms} Baths</span>
            </div>
            <div className="flex items-center">
              <Square className="h-6 w-6 mr-2" />
              <span>{property.total_area} sqft</span>
            </div>
          </div>

          <div className="mt-12">
            <EnquiryForm propertyId={property.id} />
          </div>
          {/* 🔹 AI-Generated Description */}
          {description && (
            <div className="mt-6 p-4 bg-gray-100 rounded-lg">
              <h3 className="text-lg font-semibold">AI-Generated Insights</h3>
              <p>{description}</p>
            </div>
          )}
        </div>
      </div>

      {/* 🔹 Similar Properties */}
      <div className="mt-16">
        <h2 className="text-3xl font-bold text-gray-900 mb-6">Similar Properties</h2>
        <PropertyList properties={recommendations} />
      </div>
    </div>
  );
}
