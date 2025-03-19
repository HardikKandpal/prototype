import React from "react";

interface Property {
  id: number;
  title: string;
  location: string;
  price: string;
  total_area: number;
}

export function PropertyList({ properties }: { properties: Property[] }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {properties.map((property) => (
        <div key={property.id} className="border p-4 rounded-lg">
          <h3 className="text-xl font-semibold">{property.title}</h3>
          <p className="text-gray-600">{property.location}</p>
          <p className="font-medium">₹{property.price}</p>
          <p>{property.total_area} sq ft</p>
        </div>
      ))}
    </div>
  );
}
